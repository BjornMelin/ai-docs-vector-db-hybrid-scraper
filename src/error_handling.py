#!/usr/bin/env python3
"""Error handling utilities for MCP server."""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any
from typing import TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class MCPError(Exception):
    """Base MCP server error."""

    def __init__(self, message: str, error_code: str | None = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class ValidationError(MCPError):
    """Input validation error."""

    pass


class ExternalServiceError(MCPError):
    """External service error."""

    pass


class RateLimitError(ExternalServiceError):
    """Rate limiting error."""

    pass


class NetworkError(ExternalServiceError):
    """Network-related error."""

    pass


class ConfigurationError(MCPError):
    """Configuration or environment error."""

    pass


def safe_response(success: bool, **kwargs) -> dict[str, Any]:
    """Create a safe response dictionary.

    Args:
        success: Whether the operation was successful
        **kwargs: Additional response data

    Returns:
        Safe response dictionary
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

        return wrapper

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
        last_failure_time = None
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
                    raise ExternalServiceError(
                        f"Circuit breaker is open for {func.__name__}. "
                        f"Try again in {recovery_timeout - (time.time() - last_failure_time):.1f}s"
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

        return wrapper

    return decorator


def handle_mcp_errors(func: F) -> F:
    """Decorator to handle MCP tool errors safely.

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
        except (
            ValidationError,
            RateLimitError,
            NetworkError,
            ExternalServiceError,
            MCPError,
        ) as e:
            error_type = {
                ValidationError: "validation",
                RateLimitError: "rate_limit",
                NetworkError: "network",
                ExternalServiceError: "external_service",
                MCPError: "mcp",
            }.get(type(e), "general")
            logger.warning(f"{error_type.capitalize()} error in {func.__name__}: {e}")
            return safe_response(False, error=str(e), error_type=error_type)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return safe_response(
                False, error="Internal server error", error_type="internal"
            )


def validate_input(**validators) -> Callable[[F], F]:
    """Decorator to validate function inputs.

    Args:
        **validators: Keyword arguments mapping parameter names to validator functions

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature
            import inspect

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
                        raise ValidationError(f"Invalid {param_name}: {e}") from e

            return await func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int = 10, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = []

    async def acquire(self) -> None:
        """Acquire rate limit permission.

        Raises:
            RateLimitError: If rate limit exceeded
        """
        now = time.time()

        # Remove old calls outside the window
        self.calls = [
            call_time
            for call_time in self.calls
            if now - call_time < self.window_seconds
        ]

        # Check if we're at the limit
        if len(self.calls) >= self.max_calls:
            raise RateLimitError(
                f"Rate limit exceeded: {self.max_calls} calls per {self.window_seconds}s"
            )

        # Record this call
        self.calls.append(now)


# Global rate limiters for different services
openai_rate_limiter = RateLimiter(
    max_calls=50, window_seconds=60
)  # 50 calls per minute
qdrant_rate_limiter = RateLimiter(
    max_calls=100, window_seconds=60
)  # 100 calls per minute
firecrawl_rate_limiter = RateLimiter(
    max_calls=20, window_seconds=60
)  # 20 calls per minute
