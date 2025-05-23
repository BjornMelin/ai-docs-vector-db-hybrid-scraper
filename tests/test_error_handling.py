#!/usr/bin/env python3
"""Tests for error handling utilities."""

import asyncio

import pytest
from src.error_handling import ExternalServiceError
from src.error_handling import MCPError
from src.error_handling import NetworkError
from src.error_handling import RateLimiter
from src.error_handling import RateLimitError
from src.error_handling import ValidationError
from src.error_handling import circuit_breaker
from src.error_handling import handle_mcp_errors
from src.error_handling import retry_async
from src.error_handling import safe_response
from src.error_handling import validate_input


class TestMCPErrors:
    """Test MCP error classes."""

    def test_mcp_error_creation(self):
        """Test MCP error creation."""
        error = MCPError("Test error", "TEST_001")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("Invalid input")
        assert isinstance(error, MCPError)
        assert str(error) == "Invalid input"

    def test_external_service_error(self):
        """Test external service error."""
        error = ExternalServiceError("Service unavailable")
        assert isinstance(error, MCPError)
        assert str(error) == "Service unavailable"


class TestSafeResponse:
    """Test safe response utility."""

    def test_success_response(self):
        """Test successful response creation."""
        response = safe_response(True, data="test", count=5)

        assert response["success"] is True
        assert response["data"] == "test"
        assert response["count"] == 5
        assert "timestamp" in response

    def test_error_response_with_string(self):
        """Test error response with string error."""
        response = safe_response(False, error="Something went wrong")

        assert response["success"] is False
        assert response["error"] == "Something went wrong"
        assert response["error_type"] == "general"
        assert "timestamp" in response

    def test_error_response_with_exception(self):
        """Test error response with exception."""
        exception = ValueError("Invalid value")
        response = safe_response(False, error=exception, error_type="validation")

        assert response["success"] is False
        assert response["error"] == "Invalid value"
        assert response["error_type"] == "validation"

    def test_error_response_sanitization(self):
        """Test error message sanitization."""
        error_msg = "Error at /home/user/secret with api_key=secret123"
        response = safe_response(False, error=error_msg)

        assert "/home/" not in response["error"]
        assert "api_key" not in response["error"]
        assert "***" in response["error"]


class TestRetryDecorator:
    """Test retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0

        @retry_async(max_attempts=3)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test successful execution after failures."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_all_attempts_fail(self):
        """Test all retry attempts fail."""
        call_count = 0

        @retry_async(max_attempts=2, base_delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Persistent failure")

        with pytest.raises(NetworkError, match="Persistent failure"):
            await test_func()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_non_retryable_error(self):
        """Test non-retryable error passes through immediately."""
        call_count = 0

        @retry_async(max_attempts=3, exceptions=(NetworkError,))
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Not retryable")

        with pytest.raises(ValidationError):
            await test_func()

        assert call_count == 1


class TestCircuitBreaker:
    """Test circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        call_count = 0

        @circuit_breaker(failure_threshold=3, recovery_timeout=0.1)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Service down")

        # First two calls should fail normally
        with pytest.raises(NetworkError):
            await test_func()

        with pytest.raises(NetworkError):
            await test_func()

        # Third call should be blocked by circuit breaker
        with pytest.raises(ExternalServiceError, match="Circuit breaker is open"):
            await test_func()

        assert call_count == 2  # Third call was blocked

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        call_count = 0
        failure_mode = True

        @circuit_breaker(failure_threshold=1, recovery_timeout=0.05)
        async def test_func():
            nonlocal call_count, failure_mode
            call_count += 1
            if failure_mode:
                raise NetworkError("Service down")
            return "success"

        # Trigger circuit breaker
        with pytest.raises(NetworkError):
            await test_func()

        # Should be blocked
        with pytest.raises(ExternalServiceError):
            await test_func()

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Fix the service
        failure_mode = False

        # Should succeed and close circuit
        result = await test_func()
        assert result == "success"


class TestHandleMCPErrors:
    """Test MCP error handling decorator."""

    @pytest.mark.asyncio
    async def test_handle_successful_function(self):
        """Test handling successful function."""

        @handle_mcp_errors
        async def test_func():
            return {"data": "test"}

        result = await test_func()
        assert result["success"] is True
        assert result["result"]["data"] == "test"

    @pytest.mark.asyncio
    async def test_handle_function_returning_response(self):
        """Test function that returns response dict."""

        @handle_mcp_errors
        async def test_func():
            return {"success": True, "message": "done"}

        result = await test_func()
        assert result["success"] is True
        assert result["message"] == "done"

    @pytest.mark.asyncio
    async def test_handle_validation_error(self):
        """Test handling validation error."""

        @handle_mcp_errors
        async def test_func():
            raise ValidationError("Invalid input")

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Invalid input"
        assert result["error_type"] == "validation"

    @pytest.mark.asyncio
    async def test_handle_external_service_error(self):
        """Test handling external service error."""

        @handle_mcp_errors
        async def test_func():
            raise ExternalServiceError("Service unavailable")

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Service unavailable"
        assert result["error_type"] == "external_service"

    @pytest.mark.asyncio
    async def test_handle_unexpected_error(self):
        """Test handling unexpected error."""

        @handle_mcp_errors
        async def test_func():
            raise ValueError("Unexpected error")

        result = await test_func()
        assert result["success"] is False
        assert result["error"] == "Internal server error"
        assert result["error_type"] == "internal"


class TestValidateInput:
    """Test input validation decorator."""

    @pytest.mark.asyncio
    async def test_validate_input_success(self):
        """Test successful input validation."""

        @validate_input(name=lambda x: x.upper(), count=lambda x: int(x))
        async def test_func(name: str, count: int):
            return f"{name}_{count}"

        result = await test_func("test", "5")
        assert result == "TEST_5"

    @pytest.mark.asyncio
    async def test_validate_input_validation_error(self):
        """Test validation error."""

        def validate_positive(x):
            if int(x) <= 0:
                raise ValueError("Must be positive")
            return int(x)

        @validate_input(count=validate_positive)
        async def test_func(count: int):
            return count

        with pytest.raises(ValidationError, match="Invalid count"):
            await test_func("-5")


class TestRateLimiter:
    """Test rate limiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_under_limit(self):
        """Test rate limiter allows calls under limit."""
        limiter = RateLimiter(max_calls=3, window_seconds=1.0)

        # Should allow first 3 calls
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self):
        """Test rate limiter blocks calls over limit."""
        limiter = RateLimiter(max_calls=2, window_seconds=1.0)

        # First 2 calls should work
        await limiter.acquire()
        await limiter.acquire()

        # Third call should be blocked
        with pytest.raises(RateLimitError):
            await limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiter_window_reset(self):
        """Test rate limiter window reset."""
        limiter = RateLimiter(max_calls=1, window_seconds=0.1)

        # First call should work
        await limiter.acquire()

        # Second call should be blocked
        with pytest.raises(RateLimitError):
            await limiter.acquire()

        # Wait for window reset
        await asyncio.sleep(0.15)

        # Should work again
        await limiter.acquire()
