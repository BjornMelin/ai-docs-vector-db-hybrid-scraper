"""Error classes for all services and MCP server.

This module provides an error hierarchy for the AI Documentation Vector DB project,
following best practices from Pydantic 2.0 and FastMCP 2.0.

Error Hierarchy:
    BaseError: Root exception with error_code, message, and context
    ├── ServiceError: Base for service-layer errors
    │   ├── QdrantServiceError: Vector database errors
    │   ├── EmbeddingServiceError: Embedding generation errors
    │   ├── CrawlServiceError: Web crawling errors
    │   └── CacheServiceError: Caching layer errors
    ├── ValidationError: Input validation errors (Pydantic integration)
    ├── MCPError: MCP server-specific errors
    │   ├── ToolError: MCP tool execution errors
    │   └── ResourceError: MCP resource access errors
    ├── APIError: External API integration errors
    │   ├── RateLimitError: Rate limiting errors
    │   ├── NetworkError: Network connectivity errors
    │   └── ExternalServiceError: General external service errors
    └── ConfigurationError: Configuration and environment errors
"""

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar, LiteralString, TypeVar

from pydantic import ValidationError as PydanticValidationError
from pydantic_core import PydanticCustomError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# Base Error Classes
class BaseError(Exception):
    """Base error class with context support."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize base error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for categorization
            context: Additional context information
        """

        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for API responses."""

        return {
            "error": self.message,
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "context": self.context,
        }


# Service Layer Errors
class ServiceError(BaseError):
    """Base class for service-layer errors."""


class QdrantServiceError(ServiceError):
    """Qdrant vector database errors."""


class EmbeddingServiceError(ServiceError):
    """Embedding service errors."""


class CrawlServiceError(ServiceError):
    """Web crawling service errors."""


class CacheServiceError(ServiceError):
    """Cache service errors."""


# Validation Errors
class ValidationError(BaseError):
    """Input validation error with Pydantic integration."""

    @classmethod
    def from_pydantic(cls, exc: PydanticValidationError) -> "ValidationError":
        """Create ValidationError from Pydantic ValidationError."""

        errors = exc.errors()
        if len(errors) == 1:
            error = errors[0]
            return cls(
                message=error["msg"],
                error_code="validation_error",
                context={
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "type": error["type"],
                    "input": error.get("input"),
                },
            )
        return cls(
            message="Multiple validation errors occurred",
            error_code="validation_error",
            context={"errors": errors},
        )


# MCP Server Errors (FastMCP pattern)
class MCPError(BaseError):
    """Base MCP server error."""


class ToolError(MCPError):
    """MCP tool execution error.

    Following FastMCP pattern where ToolError contents are sent to clients.
    """


class ResourceError(MCPError):
    """MCP resource access error.

    Following FastMCP pattern where ResourceError contents are sent to clients.
    """


# API Integration Errors
class APIError(BaseError):
    """Base error for external API integrations."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize API error with HTTP status code."""
        super().__init__(message, error_code, context)
        self.status_code = status_code


class ExternalServiceError(APIError):
    """General external service error."""


class RateLimitError(ExternalServiceError):
    """Rate limiting error with retry information."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            error_code: Error code
            context: Additional context

        """
        super().__init__(message, 429, error_code, context)
        self.retry_after = retry_after
        if retry_after:
            self.context["retry_after"] = retry_after


class NetworkError(ExternalServiceError):
    """Network connectivity error."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize network error."""
        super().__init__(message, 503, error_code, context)


class ConfigurationError(BaseError):
    """Configuration or environment error."""


# Utility Functions and Decorators
def safe_response(success: bool, **kwargs) -> dict[str, Any]:
    """Create a safe response dictionary for MCP tools.

    Args:
        success: Whether the operation was successful
        **kwargs: Additional response data

    Returns:
        Safe response dictionary with sanitized error messages

    """
    response = {"success": success, "timestamp": time.time()}

    if success:
        response.update(kwargs)
    else:
        # Sanitize error messages
        error = kwargs.get("error", "Unknown error")
        if isinstance(error, Exception):
            error = str(error)

        # Don't expose internal paths or sensitive info
        error = error.replace("/home/", "/****/")
        error = error.replace("api_key", "***")
        error = error.replace("token", "***")
        error = error.replace("password", "***")
        error = error.replace("secret", "***")

        response["error"] = error
        response["error_type"] = kwargs.get("error_type", "general")

    return response


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Async retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between attempts in seconds
        max_delay: Maximum delay between attempts in seconds
        backoff_factor: Factor to increase delay by each attempt
        exceptions: Exception types to retry on

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.exception("Final attempt failed for %s", func.__name__)
                        break

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                        attempt + 1,
                        max_attempts,
                        func.__name__,
                        e,
                        delay,
                    )

                    await asyncio.sleep(delay)
                except (TimeoutError, OSError, PermissionError):
                    # Non-retryable error
                    logger.exception("Non-retryable error in %s", func.__name__)
                    raise

            raise last_exception or Exception("All retry attempts failed")

        return wrapper  # type: ignore[misc]

    return decorator


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerMetrics:  # pylint: disable=too-many-instance-attributes
    """Metrics collection for circuit breaker monitoring."""

    def __init__(self):
        self.total_calls = 0
        self.success_calls = 0
        self.failure_calls = 0
        self.circuit_opens = 0
        self.circuit_closes = 0
        self.state_transitions = []
        self.response_times = []
        self.last_reset = datetime.now(tz=UTC)

    def record_call(self, success: bool, response_time: float | None = None):
        """Record a service call."""

        self.total_calls += 1
        if success:
            self.success_calls += 1
        else:
            self.failure_calls += 1

        if response_time is not None:
            self.response_times.append(response_time)

    def record_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Record a circuit state transition."""

        self.state_transitions.append(
            {
                "timestamp": datetime.now(tz=UTC),
                "from_state": old_state.value,
                "to_state": new_state.value,
            }
        )

        if new_state == CircuitState.OPEN:
            self.circuit_opens += 1
        elif new_state == CircuitState.CLOSED:
            self.circuit_closes += 1

    def get_success_rate(self) -> float:
        """Get the success rate percentage."""

        if self.total_calls == 0:
            return 100.0
        return (self.success_calls / self.total_calls) * 100

    def get_average_response_time(self) -> float:
        """Get average response time in seconds."""

        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def reset(self):
        """Reset all metrics."""
        self.total_calls = 0
        self.success_calls = 0
        self.failure_calls = 0
        self.circuit_opens = 0
        self.circuit_closes = 0
        self.state_transitions = []
        self.response_times = []
        self.last_reset = datetime.now(tz=UTC)


class AdvancedCircuitBreaker:  # pylint: disable=too-many-instance-attributes
    """Circuit breaker with adaptive features and metrics."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
        enable_adaptive_timeout: bool = True,
        enable_metrics: bool = True,
    ):
        """Initialize circuit breaker."""

        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.expected_exceptions = expected_exceptions
        self.enable_adaptive_timeout = enable_adaptive_timeout
        self.enable_metrics = enable_metrics

        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time: datetime | None = None
        self.last_success_time: datetime | None = None

        # Adaptive timeout
        self.adaptive_timeout = recovery_timeout
        self.consecutive_successes = 0

        # Metrics
        self.metrics = CircuitBreakerMetrics() if enable_metrics else None

        # Global registry for monitoring
        CircuitBreakerRegistry.register(service_name, self)

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should transition from OPEN to HALF_OPEN."""

        if self.state != CircuitState.OPEN:
            return False

        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now(tz=UTC) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.adaptive_timeout

    def should_attempt_reset(self) -> bool:
        """Check if circuit should transition from OPEN to HALF_OPEN (public interface).

        Returns:
            True if circuit should attempt to reset
        """

        return self._should_attempt_reset()

    def record_success(self, response_time: float):
        """Record a successful call (public interface).

        Args:
            response_time: Response time in seconds
        """

        return self._record_success(response_time)

    def record_failure(self, exception: Exception, response_time: float):
        """Record a failed call (public interface).

        Args:
            exception: The exception that occurred
            response_time: Response time in seconds
        """

        return self._record_failure(exception, response_time)

    def _update_adaptive_timeout(self, success: bool):
        """Update adaptive timeout based on recent success/failure patterns."""

        if not self.enable_adaptive_timeout:
            return

        if success:
            self.consecutive_successes += 1
            # Reduce timeout after consecutive successes
            if self.consecutive_successes >= 3:
                self.adaptive_timeout = max(
                    self.recovery_timeout * 0.7, self.recovery_timeout / 4
                )
        else:
            self.consecutive_successes = 0
            # Increase timeout after failures
            self.adaptive_timeout = min(
                self.recovery_timeout * 1.5, self.recovery_timeout * 3
            )

    def _change_state(self, new_state: CircuitState):
        """Change circuit state and record metrics."""

        old_state = self.state
        self.state = new_state

        if self.metrics:
            self.metrics.record_state_change(old_state, new_state)

        logger.info(
            "Circuit breaker '%s' state changed: %s -> %s",
            self.service_name,
            old_state.value,
            new_state.value,
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""

        start_time = time.time()

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._change_state(CircuitState.HALF_OPEN)
                self.half_open_calls = 0
            else:
                wait_time = (
                    self.adaptive_timeout
                    - (datetime.now(tz=UTC) - self.last_failure_time).total_seconds()
                    if self.last_failure_time
                    else self.adaptive_timeout
                )
                msg = (
                    f"Circuit breaker is open for {self.service_name}. "
                    f"Try again in {wait_time:.1f}s"
                )
                raise ExternalServiceError(
                    msg,
                    context={"service": self.service_name, "state": self.state.value},
                )

        # Check half-open call limit
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                msg = (
                    f"Circuit breaker is half-open for {self.service_name}. "
                    f"Maximum test calls ({self.half_open_max_calls}) exceeded."
                )
                raise ExternalServiceError(
                    msg,
                    context={"service": self.service_name, "state": self.state.value},
                )
            self.half_open_calls += 1

        try:
            # Execute the function
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )
        except self.expected_exceptions as e:
            # Record failure
            response_time = time.time() - start_time
            self._record_failure(e, response_time)
            raise
        except Exception as e:
            # Non-retryable error - don't count against circuit breaker
            logger.warning(
                "Non-retryable error in %s: %s. Not counted against circuit breaker.",
                self.service_name,
                e,
            )
            raise

        # Record success
        response_time = time.time() - start_time
        self._record_success(response_time)
        return result

    def _record_success(self, response_time: float):
        """Record a successful call."""
        self.last_success_time = datetime.now(tz=UTC)
        self._update_adaptive_timeout(success=True)

        if self.metrics:
            self.metrics.record_call(success=True, response_time=response_time)

        # Handle state transitions on success
        if self.state == CircuitState.HALF_OPEN:
            # Successful call in half-open state - close circuit
            self._change_state(CircuitState.CLOSED)
            self.failure_count = 0
            self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            if self.failure_count > 0:
                self.failure_count = 0
                logger.debug(
                    "Circuit breaker '%s' failure count reset", self.service_name
                )

    def _record_failure(self, _exception: Exception, response_time: float):
        """Record a failed call."""

        self.failure_count += 1
        self.last_failure_time = datetime.now(tz=UTC)
        self._update_adaptive_timeout(success=False)

        if self.metrics:
            self.metrics.record_call(success=False, response_time=response_time)

        logger.warning(
            "Circuit breaker '%s' failure %d/%d",
            self.service_name,
            self.failure_count,
            self.failure_threshold,
        )

        # Check if we should open the circuit
        if (
            self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]
            and self.failure_count >= self.failure_threshold
        ):
            self._change_state(CircuitState.OPEN)
            self.half_open_calls = 0

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""

        status = {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "adaptive_timeout": self.adaptive_timeout,
            "last_failure_time": self.last_failure_time.isoformat()
            if self.last_failure_time
            else None,
            "last_success_time": self.last_success_time.isoformat()
            if self.last_success_time
            else None,
        }

        if self.metrics:
            status.update(
                {
                    "total_calls": self.metrics.total_calls,
                    "success_rate": self.metrics.get_success_rate(),
                    "avg_response_time": self.metrics.get_average_response_time(),
                    "circuit_opens": self.metrics.circuit_opens,
                    "circuit_closes": self.metrics.circuit_closes,
                }
            )

        return status

    def reset(self):
        """Manually reset the circuit breaker."""

        self._change_state(CircuitState.CLOSED)
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        self.adaptive_timeout = self.recovery_timeout

        if self.metrics:
            self.metrics.reset()

        logger.info("Circuit breaker '%s' manually reset", self.service_name)


class CircuitBreakerRegistry:
    """Global registry for circuit breaker instances."""

    _breakers: ClassVar[dict[str, AdvancedCircuitBreaker]] = {}

    @classmethod
    def register(cls, service_name: str, breaker: AdvancedCircuitBreaker):
        """Register a circuit breaker."""
        cls._breakers[service_name] = breaker

    @classmethod
    def get(cls, service_name: str) -> AdvancedCircuitBreaker | None:
        """Get a circuit breaker by service name."""
        return cls._breakers.get(service_name)

    @classmethod
    def get_all_status(cls) -> dict[str, dict[str, Any]]:
        """Get status of all registered circuit breakers."""
        return {name: breaker.get_status() for name, breaker in cls._breakers.items()}

    @classmethod
    def reset_all(cls):
        """Reset all circuit breakers."""
        for breaker in cls._breakers.values():
            breaker.reset()

    @classmethod
    def get_services(cls) -> list[str]:
        """Get list of all registered service names."""
        return list(cls._breakers.keys())


def circuit_breaker(  # pylint: disable=too-many-arguments
    service_name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_max_calls: int = 3,
    expected_exceptions: tuple[type[Exception], ...] | None = None,
    *,
    expected_exception: type[Exception] | None = None,
    enable_adaptive_timeout: bool = True,
    enable_metrics: bool = True,
) -> Callable[[F], F]:
    """Advanced circuit breaker decorator with modern features.

    Args:
        service_name: Name of the service (defaults to function name)
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Base time to wait before attempting recovery
        half_open_max_calls: Maximum calls allowed in half-open state
        expected_exceptions: Exception types that trigger circuit breaker (tuple)
        expected_exception: Backwards-compatible singular exception to monitor
        enable_adaptive_timeout: Enable adaptive timeout adjustment
        enable_metrics: Enable metrics collection

    Returns:
        Decorated function with circuit breaker protection
    """

    def decorator(func: F) -> F:
        nonlocal service_name
        if service_name is None:
            service_name = f"{func.__module__}.{func.__name__}"

        monitored_exceptions = expected_exceptions or (Exception,)
        if (
            expected_exception is not None
            and expected_exception not in monitored_exceptions
        ):
            monitored_exceptions = (expected_exception, *monitored_exceptions)

        # Create or get existing circuit breaker for this service
        breaker = CircuitBreakerRegistry.get(service_name)
        if breaker is None:
            breaker = AdvancedCircuitBreaker(
                service_name=service_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
                expected_exceptions=monitored_exceptions,
                enable_adaptive_timeout=enable_adaptive_timeout,
                enable_metrics=enable_metrics,
            )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        # Add circuit breaker management methods to the wrapper
        wrapper.circuit_breaker = breaker  # type: ignore[misc]
        wrapper.get_circuit_status = breaker.get_status  # type: ignore[misc]
        wrapper.reset_circuit = breaker.reset  # type: ignore[misc]

        return wrapper  # type: ignore[misc]

    return decorator


def tenacity_circuit_breaker(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    service_name: str | None = None,
    max_attempts: int = 3,
    wait_multiplier: float = 1.0,
    wait_max: float = 10.0,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exceptions: tuple[type[Exception], ...] = (
        ExternalServiceError,
        NetworkError,
        RateLimitError,
    ),
) -> Callable[[F], F]:
    """Tenacity-powered circuit breaker with exponential backoff.

    Combines Tenacity's retry capabilities with circuit breaker pattern.

    Args:
        service_name: Name of the service
        max_attempts: Maximum retry attempts
        wait_multiplier: Exponential backoff multiplier
        wait_max: Maximum wait time between retries
        failure_threshold: Failures before opening circuit
        recovery_timeout: Circuit recovery timeout
        expected_exceptions: Exceptions to retry on

    Returns:
        Decorated function with retry + circuit breaker protection
    """

    def decorator(func: F) -> F:
        nonlocal service_name
        if service_name is None:
            service_name = f"{func.__module__}.{func.__name__}"

        # Create circuit breaker
        breaker = AdvancedCircuitBreaker(
            service_name=f"{service_name}_tenacity",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions,
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check circuit breaker state first
            if (
                breaker.state == CircuitState.OPEN
                and not breaker.should_attempt_reset()
            ):
                wait_time = (
                    breaker.adaptive_timeout
                    - (datetime.now(tz=UTC) - breaker.last_failure_time).total_seconds()
                    if breaker.last_failure_time
                    else breaker.adaptive_timeout
                )
                msg = (
                    f"Circuit breaker is OPEN for {service_name}. "
                    f"Try again in {wait_time:.1f}s"
                )
                raise ExternalServiceError(msg)

            # Use Tenacity for retries with circuit breaker integration
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=wait_multiplier, max=wait_max),
                retry=retry_if_exception_type(expected_exceptions),
                reraise=True,
            ):
                with attempt:
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        response_time = time.time() - start_time
                        breaker.record_success(response_time)
                    except expected_exceptions as e:
                        response_time = time.time() - start_time
                        breaker.record_failure(e, response_time)

                        # Log retry attempt
                        logger.warning(
                            "Tenacity retry attempt %d/%d failed for %s",
                            attempt.retry_state.attempt_number,
                            max_attempts,
                            service_name,
                        )
                        raise

                    return result
            return None

        # Add circuit breaker management methods
        wrapper.circuit_breaker = breaker  # type: ignore[misc]
        wrapper.get_circuit_status = breaker.get_status  # type: ignore[misc]
        wrapper.reset_circuit = breaker.reset  # type: ignore[misc]

        return wrapper  # type: ignore[misc]

    return decorator


def handle_mcp_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to handle MCP tool errors safely.

    Following FastMCP patterns where ToolError and ResourceError
    contents are sent to clients while other exceptions are masked.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that returns safe responses
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> dict[str, Any]:
        try:
            result = await func(*args, **kwargs)
            if isinstance(result, dict) and "success" in result:
                return result
            return safe_response(True, result=result)
        except (ToolError, ResourceError) as e:
            # These errors are meant to be sent to clients
            logger.warning("MCP error in %s: %s", func.__name__, e)
            return safe_response(
                False, error=str(e), error_type=e.error_code or "mcp_error"
            )
        except (
            ValidationError,
            RateLimitError,
            NetworkError,
            ExternalServiceError,
            ConfigurationError,
        ) as e:
            error_type_map = {
                ValidationError: "validation",
                RateLimitError: "rate_limit",
                NetworkError: "network",
                ExternalServiceError: "external_service",
                ConfigurationError: "configuration",
            }
            error_type = error_type_map.get(type(e), "general")
            logger.warning(
                "%s error in %s: %s", error_type.capitalize(), func.__name__, e
            )
            return safe_response(False, error=str(e), error_type=error_type)
        except (ConnectionError, OSError, PermissionError):
            # Mask internal errors for security
            logger.exception("Unexpected error in %s", func.__name__)
            return safe_response(
                False, error="Internal server error", error_type="internal"
            )
        except Exception:
            logger.exception("Unexpected error in %s", func.__name__)
            return safe_response(
                False, error="Internal server error", error_type="internal"
            )

    return wrapper  # type: ignore[misc]


def validate_input(**validators) -> Callable[[F], F]:
    """Decorator to validate function inputs using Pydantic-style validation.

    Args:
        **validators: Keyword arguments mapping parameter names to validator functions

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validated_value = validator(value)
                        bound_args.arguments[param_name] = validated_value
                    except Exception as e:  # pragma: no cover - validator bug
                        msg = f"Invalid {param_name}: {e}"
                        raise ValidationError(
                            msg,
                            error_code="invalid_input",
                            context={"field": param_name, "value": value},
                        ) from e

            return await func(*bound_args.args, **bound_args.kwargs)

        return wrapper  # type: ignore[misc]

    return decorator


# Custom Pydantic errors following Pydantic 2.0 patterns
def create_validation_error(
    field: str,
    message: LiteralString,
    error_type: LiteralString = "value_error",
    **context: Any,
) -> PydanticCustomError:
    """Create a custom Pydantic validation error.

    Args:
        field: Field name that failed validation
        message: Error message (must be a literal string)
        error_type: Type of validation error (must be a literal string)
        **context: Additional context for the error

    Returns:
        PydanticCustomError instance
    """
    return PydanticCustomError(
        error_type,
        message,
        {"field": field, **context},
    )
