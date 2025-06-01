"""Decorators and patterns for the AI Documentation Vector DB system.

This module provides common decorators for retry logic, circuit breakers,
error handling, rate limiting, and input validation.
"""

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from .errors import ConfigurationError
from .errors import ExternalServiceError
from .errors import NetworkError
from .errors import RateLimitError
from .errors import ResourceError
from .errors import ToolError
from .errors import ValidationError
from .errors import safe_response

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


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
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        break

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            raise last_exception or Exception("All retry attempts failed")

        return wrapper  # type: ignore

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
) -> Callable[[F], F]:
    """Circuit breaker pattern decorator.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        failure_count = 0
        last_failure_time: float | None = None
        state = "closed"  # closed, open, half-open

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, state

            # Check if circuit should be half-open
            if state == "open":
                if (
                    last_failure_time
                    and time.time() - last_failure_time > recovery_timeout
                ):
                    state = "half-open"
                    logger.info(f"Circuit breaker for {func.__name__} is half-open")
                else:
                    wait_time = (
                        recovery_timeout - (time.time() - last_failure_time)
                        if last_failure_time
                        else recovery_timeout
                    )
                    raise ExternalServiceError(
                        f"Circuit breaker is open for {func.__name__}. "
                        f"Try again in {wait_time:.1f}s"
                    )

            try:
                result = await func(*args, **kwargs)

                # Reset on success
                if state == "half-open":
                    state = "closed"
                    failure_count = 0
                    logger.info(
                        f"Circuit breaker for {func.__name__} closed (recovered)"
                    )

                return result

            except expected_exception as e:
                failure_count += 1
                last_failure_time = time.time()

                logger.warning(
                    f"Circuit breaker failure {failure_count}/{failure_threshold} "
                    f"for {func.__name__}: {e}"
                )

                if failure_count >= failure_threshold:
                    state = "open"
                    logger.error(f"Circuit breaker opened for {func.__name__}")

                raise

        return wrapper  # type: ignore

    return decorator


def handle_mcp_errors(func: F) -> F:
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
            logger.warning(f"MCP error in {func.__name__}: {e}")
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
            logger.warning(f"{error_type.capitalize()} error in {func.__name__}: {e}")
            return safe_response(False, error=str(e), error_type=error_type)
        except Exception as e:
            # Mask internal errors for security
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return safe_response(
                False, error="Internal server error", error_type="internal"
            )

    return wrapper  # type: ignore


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
                    except Exception as e:
                        raise ValidationError(
                            f"Invalid {param_name}: {e}",
                            error_code="invalid_input",
                            context={"field": param_name, "value": value},
                        ) from e

            return await func(*bound_args.args, **bound_args.kwargs)

        return wrapper  # type: ignore

    return decorator


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int = 10, window_seconds: float = 60.0):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire rate limit permission.

        Raises:
            RateLimitError: If rate limit exceeded
        """
        async with self._lock:
            now = time.time()

            # Remove old calls outside the window
            self.calls = [
                call_time
                for call_time in self.calls
                if now - call_time < self.window_seconds
            ]

            # Check if we're at the limit
            if len(self.calls) >= self.max_calls:
                # Calculate when the oldest call will expire
                retry_after = self.window_seconds - (now - self.calls[0])
                raise RateLimitError(
                    f"Rate limit exceeded: {self.max_calls} calls per {self.window_seconds}s",
                    retry_after=retry_after,
                )

            # Record this call
            self.calls.append(now)


__all__ = [
    "RateLimiter",
    "circuit_breaker",
    "handle_mcp_errors",
    "retry_async",
    "validate_input",
]
