"""Unit tests for core decorators module."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.services.errors import (
    ExternalServiceError,
    RateLimitError,
    ResourceError,
    ToolError,
    ValidationError,
    circuit_breaker,
    handle_mcp_errors,
    retry_async,
    validate_input,
)
from src.services.utilities.rate_limiter import RateLimiter


class TestRetryAsync:
    """Test cases for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test retry decorator when function succeeds on first attempt."""
        mock_func = AsyncMock(return_value="success")

        @retry_async(max_attempts=3)
        @pytest.mark.asyncio
        async def test_func():
            return await mock_func()

        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self):
        """Test retry decorator when function succeeds after retries."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "Temporary error"
                raise ValueError(msg)
            return "success"

        @retry_async(max_attempts=3, base_delay=0.01)  # Fast for testing
        @pytest.mark.asyncio
        async def test_func():
            return await flaky_func()

        result = await test_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_all_attempts_fail(self):
        """Test retry decorator when all attempts fail."""
        mock_func = AsyncMock(side_effect=ValueError("Persistent error"))

        @retry_async(max_attempts=3, base_delay=0.01)
        @pytest.mark.asyncio
        async def test_func():
            return await mock_func()

        with pytest.raises(ValueError, match="Persistent error"):
            await test_func()

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_non_retryable_exception(self):
        """Test retry decorator with non-retryable exception."""

        @retry_async(max_attempts=3, exceptions=(ValueError,))
        @pytest.mark.asyncio
        async def test_func():
            msg = "Non-retryable error"
            raise TypeError(msg)

        with pytest.raises(TypeError, match="Non-retryable error"):
            await test_func()

    @pytest.mark.asyncio
    async def test_retry_async_backoff_calculation(self):
        """Test that backoff delay increases correctly."""
        delays = []
        original_sleep = asyncio.sleep

        async def capture_sleep(delay):
            delays.append(delay)
            # Don't actually sleep to keep test fast

        # Mock asyncio.sleep to capture delays
        asyncio.sleep = capture_sleep

        try:

            @retry_async(max_attempts=3, base_delay=1.0, backoff_factor=2.0)
            @pytest.mark.asyncio
            async def test_func():
                msg = "Test error"
                raise ValueError(msg)

            with pytest.raises(ValueError):
                await test_func()

            # Should have 2 delays (attempts 1 and 2, no delay after final attempt)
            assert len(delays) == 2
            assert delays[0] == 1.0  # base_delay * (backoff_factor ** 0)
            assert delays[1] == 2.0  # base_delay * (backoff_factor ** 1)
        finally:
            asyncio.sleep = original_sleep

    @pytest.mark.asyncio
    async def test_retry_async_max_delay_respected(self):
        """Test that maximum delay is respected."""
        delays = []
        original_sleep = asyncio.sleep

        async def capture_sleep(delay):
            delays.append(delay)

        asyncio.sleep = capture_sleep

        try:

            @retry_async(
                max_attempts=4, base_delay=10.0, max_delay=15.0, backoff_factor=2.0
            )
            @pytest.mark.asyncio
            async def test_func():
                msg = "Test error"
                raise ValueError(msg)

            with pytest.raises(ValueError):
                await test_func()

            # All delays should be capped at max_delay
            for delay in delays:
                assert delay <= 15.0
        finally:
            asyncio.sleep = original_sleep


class TestCircuitBreaker:
    """Test cases for circuit_breaker decorator."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        mock_func = AsyncMock(return_value="success")

        @circuit_breaker(failure_threshold=3)
        @pytest.mark.asyncio
        async def test_func():
            return await mock_func()

        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        @pytest.mark.asyncio
        async def test_func():
            msg = "Test error"
            raise ValueError(msg)

        # First two failures should pass through
        with pytest.raises(ValueError):
            await test_func()
        with pytest.raises(ValueError):
            await test_func()

        # Third call should be blocked by open circuit
        with pytest.raises(ExternalServiceError, match="Circuit breaker is open"):
            await test_func()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                msg = "Initial failures"
                raise ValueError(msg)
            return "recovered"

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.05)
        @pytest.mark.asyncio
        async def test_func():
            return await flaky_func()

        # Trigger circuit breaker opening
        with pytest.raises(ValueError):
            await test_func()
        with pytest.raises(ValueError):
            await test_func()

        # Circuit should be open
        with pytest.raises(ExternalServiceError):
            await test_func()

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Should be half-open and allow one attempt, which succeeds
        result = await test_func()
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker behavior when half-open attempt fails."""

        @circuit_breaker(failure_threshold=1, recovery_timeout=0.05)
        @pytest.mark.asyncio
        async def test_func():
            msg = "Still failing"
            raise ValueError(msg)

        # Trigger circuit opening
        with pytest.raises(ValueError):
            await test_func()

        # Wait for recovery
        await asyncio.sleep(0.1)

        # Half-open attempt should fail and re-open circuit
        with pytest.raises(ValueError):
            await test_func()

        # Should be open again
        with pytest.raises(ExternalServiceError):
            await test_func()


class TestHandleMCPErrors:
    """Test cases for handle_mcp_errors decorator."""

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_success(self):
        """Test MCP error handler with successful function."""

        @handle_mcp_errors
        @pytest.mark.asyncio
        async def test_func():
            return {"data": "success"}

        result = await test_func()
        assert result["success"] is True
        assert result["result"]["data"] == "success"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_tool_error(self):
        """Test MCP error handler with ToolError."""

        @handle_mcp_errors
        @pytest.mark.asyncio
        async def test_func():
            msg = "Tool execution failed"
            raise ToolError(msg, error_code="tool_error")

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Tool execution failed"
        assert result["error_type"] == "tool_error"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_resource_error(self):
        """Test MCP error handler with ResourceError."""

        @handle_mcp_errors
        @pytest.mark.asyncio
        async def test_func():
            msg = "Resource not found"
            raise ResourceError(msg, error_code="not_found")

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Resource not found"
        assert result["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_validation_error(self):
        """Test MCP error handler with ValidationError."""

        @handle_mcp_errors
        @pytest.mark.asyncio
        async def test_func():
            msg = "Invalid input"
            raise ValidationError(msg, error_code="validation_failed")

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Invalid input"
        assert result["error_type"] == "validation"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_rate_limit_error(self):
        """Test MCP error handler with RateLimitError."""

        @handle_mcp_errors
        @pytest.mark.asyncio
        async def test_func():
            msg = "Rate limit exceeded"
            raise RateLimitError(msg, retry_after=60.0)

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Rate limit exceeded"
        assert result["error_type"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_generic_exception(self):
        """Test MCP error handler with generic exception."""

        @handle_mcp_errors
        @pytest.mark.asyncio
        async def test_func():
            msg = "Unexpected error"
            raise RuntimeError(msg)

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Internal server error"  # Masked for security
        assert result["error_type"] == "internal"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_preserves_safe_response(self):
        """Test that existing safe response format is preserved."""

        @handle_mcp_errors
        @pytest.mark.asyncio
        async def test_func():
            return {"success": True, "data": "custom", "timestamp": 123456}

        result = await test_func()
        assert result["success"] is True
        assert result["data"] == "custom"
        assert result["timestamp"] == 123456


class TestValidateInput:
    """Test cases for validate_input decorator."""

    @pytest.mark.asyncio
    async def test_validate_input_success(self):
        """Test input validation with valid inputs."""

        def validate_number(value):
            if not isinstance(value, int | float):
                msg = "Must be a number"
                raise TypeError(msg)
            return float(value)

        @validate_input(num=validate_number)
        @pytest.mark.asyncio
        async def test_func(num):
            return num * 2

        result = await test_func(5)
        assert result == 10.0

    @pytest.mark.asyncio
    async def test_validate_input_validation_failure(self):
        """Test input validation with invalid inputs."""

        def validate_positive(value):
            if value <= 0:
                msg = "Must be positive"
                raise ValueError(msg)
            return value

        @validate_input(num=validate_positive)
        @pytest.mark.asyncio
        async def test_func(num):
            return num

        with pytest.raises(ValidationError, match="Invalid num"):
            await test_func(-5)

    @pytest.mark.asyncio
    async def test_validate_input_multiple_validators(self):
        """Test input validation with multiple validators."""

        def validate_string(value):
            if not isinstance(value, str):
                msg = "Must be string"
                raise TypeError(msg)
            return value.strip()

        def validate_number(value):
            return int(value)

        @validate_input(name=validate_string, age=validate_number)
        @pytest.mark.asyncio
        async def test_func(name, age):
            return f"{name} is {age}"

        result = await test_func("  John  ", "25")
        assert result == "John is 25"

    @pytest.mark.asyncio
    async def test_validate_input_with_defaults(self):
        """Test input validation with default parameters."""

        def validate_string(value):
            return value.upper()

        @validate_input(name=validate_string)
        @pytest.mark.asyncio
        async def test_func(name="default"):
            return name

        result = await test_func()
        assert result == "DEFAULT"

    @pytest.mark.asyncio
    async def test_validate_input_missing_parameter(self):
        """Test input validation when validator parameter is missing."""

        def validate_number(value):
            return int(value)

        @validate_input(missing_param=validate_number)
        @pytest.mark.asyncio
        async def test_func(present_param):
            return present_param

        # Should not raise error if validated parameter is not present
        result = await test_func("test")
        assert result == "test"


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_within_limit(self):
        """Test rate limiter when within limits."""
        limiter = RateLimiter(max_calls=5, window_seconds=1.0)

        # Should allow calls within limit
        for _ in range(5):
            await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_exceeds_limit(self):
        """Test rate limiter when exceeding limits."""
        limiter = RateLimiter(max_calls=2, window_seconds=1.0)

        # First two calls should succeed
        await limiter.acquire()
        await limiter.acquire()

        # Third call should raise RateLimitError
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_window_reset(self):
        """Test rate limiter window reset."""
        limiter = RateLimiter(max_calls=2, window_seconds=0.1)

        # Fill up the limit
        await limiter.acquire()
        await limiter.acquire()

        # Should be at limit
        with pytest.raises(RateLimitError):
            await limiter.acquire()

        # Wait for window to reset
        await asyncio.sleep(0.15)

        # Should be able to make calls again
        await limiter.acquire()
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_retry_after(self):
        """Test rate limiter retry_after information."""
        limiter = RateLimiter(max_calls=1, window_seconds=1.0)

        # Fill up the limit
        await limiter.acquire()

        # Next call should provide retry_after
        with pytest.raises(RateLimitError) as exc_info:
            await limiter.acquire()

        e = exc_info.value
        assert hasattr(e, "retry_after")
        assert e.retry_after > 0
        assert e.retry_after <= 1.0

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_access(self):
        """Test rate limiter with concurrent access."""
        limiter = RateLimiter(max_calls=3, window_seconds=1.0)

        # Create multiple concurrent tasks
        async def make_request():
            await limiter.acquire()
            return "success"

        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have some successes and some rate limit errors
        successes = [r for r in results if r == "success"]
        errors = [r for r in results if isinstance(r, RateLimitError)]

        assert len(successes) == 3
        assert len(errors) == 2

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_calls=10, window_seconds=60.0)

        assert limiter.max_calls == 10
        assert limiter.window_seconds == 60.0
        assert limiter.calls == []
        assert hasattr(limiter, "_lock")

    def test_rate_limiter_default_values(self):
        """Test rate limiter with default values."""
        limiter = RateLimiter()

        assert limiter.max_calls == 10
        assert limiter.window_seconds == 60.0
