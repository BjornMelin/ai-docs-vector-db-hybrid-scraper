"""Tests for services/errors.py - Service-specific error handling.

This module tests the comprehensive error hierarchy and utility functions
for all services and MCP server error handling.
"""

import asyncio  # noqa: PLC0415
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.services.errors import (
    APIError,  # API errors
    BaseError,  # Base errors
    CacheServiceError,
    ConfigurationError,  # Configuration errors
    CrawlServiceError,
    EmbeddingServiceError,
    ExternalServiceError,
    MCPError,  # MCP errors
    NetworkError,
    QdrantServiceError,  # Service-specific errors
    RateLimitError,
    ResourceError,
    ServiceError,
    ToolError,
    ValidationError,  # Validation errors
    circuit_breaker,
    create_validation_error,  # Pydantic utilities
    handle_mcp_errors,
    retry_async,
    safe_response,  # Utility functions and decorators
    validate_input,
)


class TestBaseError:
    """Test cases for BaseError class."""

    def test_base_error_with_message_only(self):
        """Test BaseError with just a message."""
        error = BaseError("Test error message")

        assert error.message == "Test error message"
        assert error.error_code == "BaseError"
        assert error.context == {}
        assert str(error) == "Test error message"

    def test_base_error_with_all_parameters(self):
        """Test BaseError with all parameters."""
        context = {"key": "value", "details": "additional info"}
        error = BaseError("Custom error", error_code="CUSTOM_001", context=context)

        assert error.message == "Custom error"
        assert error.error_code == "CUSTOM_001"
        assert error.context == context

    def test_base_error_to_dict(self):
        """Test BaseError to_dict method."""
        context = {"request_id": "123", "user_id": "456"}
        error = BaseError("API error occurred", error_code="API_001", context=context)

        result = error.to_dict()

        expected = {
            "error": "API error occurred",
            "error_code": "API_001",
            "error_type": "BaseError",
            "context": context,
        }
        assert result == expected

    def test_base_error_inheritance(self):
        """Test BaseError inheritance behavior."""
        error = BaseError("Test")
        assert isinstance(error, Exception)
        assert isinstance(error, BaseError)


class TestServiceErrors:
    """Test cases for service-specific error classes."""

    def test_service_error_hierarchy(self):
        """Test ServiceError inherits from BaseError."""
        error = ServiceError("Service failed")
        assert isinstance(error, BaseError)
        assert error.error_code == "ServiceError"

    def test_qdrant_service_error(self):
        """Test QdrantServiceError."""
        error = QdrantServiceError(
            "Vector search failed", context={"collection": "documents"}
        )
        assert isinstance(error, ServiceError)
        assert error.message == "Vector search failed"
        assert error.context["collection"] == "documents"

    def test_embedding_service_error(self):
        """Test EmbeddingServiceError."""
        error = EmbeddingServiceError(
            "OpenAI API quota exceeded",
            error_code="OPENAI_QUOTA",
            context={"model": "text-embedding-ada-002"},
        )
        assert isinstance(error, ServiceError)
        assert error.error_code == "OPENAI_QUOTA"

    def test_crawl_service_error(self):
        """Test CrawlServiceError."""
        error = CrawlServiceError(
            "Website unreachable",
            context={"url": "https://example.com", "status_code": 404},
        )
        assert isinstance(error, ServiceError)
        assert error.context["url"] == "https://example.com"

    def test_cache_service_error(self):
        """Test CacheServiceError."""
        error = CacheServiceError(
            "Redis connection lost",
            error_code="REDIS_CONN",
            context={"host": "localhost", "port": 6379},
        )
        assert isinstance(error, ServiceError)
        assert error.error_code == "REDIS_CONN"


class TestValidationError:
    """Test cases for ValidationError class."""

    def test_validation_error_basic(self):
        """Test basic ValidationError creation."""
        error = ValidationError(
            "Invalid input format",
            error_code="INVALID_FORMAT",
            context={"field": "email", "value": "invalid-email"},
        )

        assert isinstance(error, BaseError)
        assert error.message == "Invalid input format"
        assert error.error_code == "INVALID_FORMAT"
        assert error.context["field"] == "email"

    def test_validation_error_from_pydantic_single_error(self):
        """Test creating ValidationError from Pydantic with single error."""
        # Mock a Pydantic ValidationError
        pydantic_error = Mock(spec=PydanticValidationError)
        pydantic_error.errors.return_value = [
            {
                "msg": "field required",
                "loc": ("user", "email"),
                "type": "missing_field",
                "input": {"user": {"name": "John"}},
            }
        ]

        error = ValidationError.from_pydantic(pydantic_error)

        assert error.message == "field required"
        assert error.error_code == "validation_error"
        assert error.context["field"] == "user.email"
        assert error.context["type"] == "missing_field"

    def test_validation_error_from_pydantic_multiple_errors(self):
        """Test creating ValidationError from Pydantic with multiple errors."""
        pydantic_error = Mock(spec=PydanticValidationError)
        pydantic_error.errors.return_value = [
            {"msg": "field required", "loc": ("email",), "type": "missing"},
            {"msg": "invalid format", "loc": ("age",), "type": "type_error"},
        ]

        error = ValidationError.from_pydantic(pydantic_error)

        assert error.message == "Multiple validation errors occurred"
        assert error.error_code == "validation_error"
        assert "errors" in error.context
        assert len(error.context["errors"]) == 2


class TestMCPErrors:
    """Test cases for MCP server error classes."""

    def test_mcp_error_hierarchy(self):
        """Test MCPError inherits from BaseError."""
        error = MCPError("MCP operation failed")
        assert isinstance(error, BaseError)
        assert error.error_code == "MCPError"

    def test_tool_error(self):
        """Test ToolError for MCP tool execution."""
        error = ToolError(
            "Search tool failed",
            error_code="SEARCH_FAILED",
            context={"query": "invalid query", "collection": "docs"},
        )
        assert isinstance(error, MCPError)
        assert error.message == "Search tool failed"
        assert error.context["query"] == "invalid query"

    def test_resource_error(self):
        """Test ResourceError for MCP resource access."""
        error = ResourceError(
            "Document not found",
            error_code="DOC_NOT_FOUND",
            context={"document_id": "abc123"},
        )
        assert isinstance(error, MCPError)
        assert error.error_code == "DOC_NOT_FOUND"


class TestAPIErrors:
    """Test cases for API integration error classes."""

    def test_api_error_basic(self):
        """Test basic APIError."""
        error = APIError("API request failed")
        assert isinstance(error, BaseError)
        assert error.message == "API request failed"
        assert error.status_code is None

    def test_api_error_with_status_code(self):
        """Test APIError with HTTP status code."""
        error = APIError(
            "Unauthorized access",
            status_code=401,
            error_code="AUTH_FAILED",
            context={"endpoint": "/api/search"},
        )
        assert error.status_code == 401
        assert error.error_code == "AUTH_FAILED"

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        error = ExternalServiceError(
            "OpenAI service unavailable",
            status_code=503,
            context={"service": "openai", "retry_count": 3},
        )
        assert isinstance(error, APIError)
        assert error.status_code == 503

    def test_rate_limit_error_basic(self):
        """Test RateLimitError without retry_after."""
        error = RateLimitError("Rate limit exceeded")
        assert isinstance(error, ExternalServiceError)
        assert error.status_code == 429
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError(
            "Too many requests",
            retry_after=60.0,
            error_code="RATE_LIMIT",
            context={"service": "openai"},
        )
        assert error.retry_after == 60.0
        assert error.context["retry_after"] == 60.0
        assert error.status_code == 429

    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError(
            "Connection timeout",
            error_code="TIMEOUT",
            context={"host": "api.openai.com", "timeout": 30},
        )
        assert isinstance(error, ExternalServiceError)
        assert error.status_code == 503
        assert error.error_code == "TIMEOUT"


class TestConfigurationError:
    """Test cases for ConfigurationError class."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Missing API key",
            error_code="MISSING_API_KEY",
            context={"service": "openai", "env_var": "OPENAI_API_KEY"},
        )
        assert isinstance(error, BaseError)
        assert error.message == "Missing API key"
        assert error.context["service"] == "openai"


class TestSafeResponse:
    """Test cases for safe_response utility function."""

    def test_safe_response_success(self):
        """Test safe_response with success=True."""
        result = safe_response(True, data="test data", count=5)

        assert result["success"] is True
        assert "timestamp" in result
        assert result["data"] == "test data"
        assert result["count"] == 5
        assert "error" not in result

    def test_safe_response_failure_with_string_error(self):
        """Test safe_response with string error."""
        result = safe_response(False, error="Something went wrong")

        assert result["success"] is False
        assert "timestamp" in result
        assert result["error"] == "Something went wrong"
        assert result["error_type"] == "general"

    def test_safe_response_failure_with_exception(self):
        """Test safe_response with Exception object."""
        error = ValueError("Invalid value provided")
        result = safe_response(False, error=error, error_type="validation")

        assert result["success"] is False
        assert result["error"] == "Invalid value provided"
        assert result["error_type"] == "validation"

    def test_safe_response_sanitizes_sensitive_data(self):
        """Test safe_response sanitizes sensitive information."""
        result = safe_response(
            False,
            error="Failed with api_key=sk-123 and token=abc and password=secret123",
        )

        error_msg = result["error"]
        assert "api_key" not in error_msg
        assert "token" not in error_msg
        assert "password" not in error_msg
        assert "***" in error_msg

    def test_safe_response_sanitizes_file_paths(self):
        """Test safe_response sanitizes file paths."""
        result = safe_response(
            False, error="File not found: /home/user/private/config.json"
        )

        assert "/home/" not in result["error"]
        assert "****/user/private/config.json" in result["error"]


class TestRetryAsync:
    """Test cases for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test retry_async with successful first attempt."""

        @retry_async(max_attempts=3)
        async def success_func():
            return "success"

        result = await success_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_async_success_after_failures(self):
        """Test retry_async with success after failures."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        with patch("asyncio.sleep") as mock_sleep:
            result = await flaky_func()

        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_async_all_attempts_fail(self):
        """Test retry_async when all attempts fail."""

        @retry_async(max_attempts=2, base_delay=0.01)
        async def failing_func():
            raise ValueError("Always fails")

        with patch("asyncio.sleep"), pytest.raises(ValueError, match="Always fails"):
            await failing_func()

    @pytest.mark.asyncio
    async def test_retry_async_exponential_backoff(self):
        """Test retry_async exponential backoff timing."""

        @retry_async(max_attempts=4, base_delay=1.0, backoff_factor=2.0)
        async def failing_func():
            raise ValueError("Error")

        with patch("asyncio.sleep") as mock_sleep, pytest.raises(ValueError):
            await failing_func()

        # Should have 3 sleep calls (for 4 attempts)
        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_retry_async_max_delay_limit(self):
        """Test retry_async respects max_delay limit."""

        @retry_async(
            max_attempts=4, base_delay=10.0, max_delay=15.0, backoff_factor=2.0
        )
        async def failing_func():
            raise ValueError("Error")

        with patch("asyncio.sleep") as mock_sleep, pytest.raises(ValueError):
            await failing_func()

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 10.0, min(20.0, 15.0), min(40.0, 15.0) = 10.0, 15.0, 15.0
        assert delays == [10.0, 15.0, 15.0]

    @pytest.mark.asyncio
    async def test_retry_async_selective_exceptions(self):
        """Test retry_async only retries specified exceptions."""

        @retry_async(max_attempts=3, exceptions=(ValueError,))
        async def selective_func(should_raise_value_error=True):
            if should_raise_value_error:
                raise ValueError("Retryable error")
            else:
                raise TypeError("Non-retryable error")

        # Should retry ValueError
        with patch("asyncio.sleep"), pytest.raises(ValueError):
            await selective_func(True)

        # Should not retry TypeError
        with pytest.raises(TypeError):
            await selective_func(False)

    @pytest.mark.asyncio
    async def test_retry_async_preserves_function_metadata(self):
        """Test retry_async preserves function metadata."""

        @retry_async()
        async def documented_func():
            """This function has documentation."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert "This function has documentation" in documented_func.__doc__


class TestCircuitBreaker:
    """Test cases for circuit_breaker decorator."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state_success(self):
        """Test circuit breaker in closed state with successful calls."""

        @circuit_breaker(failure_threshold=3)
        async def working_func():
            return "success"

        result = await working_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker counts failures correctly."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Failure {call_count}")

        # First failure
        with pytest.raises(ValueError, match="Failure 1"):
            await failing_func()

        # Second failure - should open circuit
        with pytest.raises(ValueError, match="Failure 2"):
            await failing_func()

        # Third call - circuit should be open
        with pytest.raises(ExternalServiceError, match="Circuit breaker is open"):
            await failing_func()

        assert call_count == 2  # Third call shouldn't execute the function

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        call_count = 0

        @circuit_breaker(failure_threshold=1, recovery_timeout=0.01)
        async def recovering_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Initial failure")
            return "recovered"

        # Trigger failure and open circuit
        with pytest.raises(ValueError):
            await recovering_func()

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Should be in half-open state and succeed
        result = await recovering_func()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state."""
        call_count = 0

        @circuit_breaker(failure_threshold=1, recovery_timeout=0.01)
        async def still_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Still failing {call_count}")

        # Open circuit
        with pytest.raises(ValueError):
            await still_failing_func()

        # Wait for recovery
        await asyncio.sleep(0.02)

        # Fail in half-open state
        with pytest.raises(ValueError, match="Still failing 2"):
            await still_failing_func()

        # Should be open again
        with pytest.raises(ExternalServiceError, match="Circuit breaker is open"):
            await still_failing_func()

    @pytest.mark.asyncio
    async def test_circuit_breaker_custom_exception(self):
        """Test circuit breaker with custom exception type."""

        @circuit_breaker(failure_threshold=1, expected_exception=ValueError)
        async def custom_exception_func(raise_value_error=True):
            if raise_value_error:
                raise ValueError("Monitored error")
            else:
                raise TypeError("Unmonitored error")

        # ValueError should trigger circuit breaker
        with pytest.raises(ValueError):
            await custom_exception_func(True)

        # Circuit should be open for any subsequent call
        with pytest.raises(ExternalServiceError):
            await custom_exception_func(False)


class TestHandleMCPErrors:
    """Test cases for handle_mcp_errors decorator."""

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_success(self):
        """Test handle_mcp_errors with successful execution."""

        @handle_mcp_errors
        async def success_func():
            return {"data": "success"}

        result = await success_func()
        assert result["success"] is True
        assert result["result"]["data"] == "success"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_with_dict_success(self):
        """Test handle_mcp_errors with dict containing success key."""

        @handle_mcp_errors
        async def dict_success_func():
            return {"success": True, "data": "test"}

        result = await dict_success_func()
        assert result["success"] is True
        assert result["data"] == "test"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_tool_error(self):
        """Test handle_mcp_errors with ToolError."""

        @handle_mcp_errors
        async def tool_error_func():
            raise ToolError("Tool execution failed", error_code="TOOL_FAILED")

        result = await tool_error_func()
        assert result["success"] is False
        assert result["error"] == "Tool execution failed"
        assert result["error_type"] == "TOOL_FAILED"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_resource_error(self):
        """Test handle_mcp_errors with ResourceError."""

        @handle_mcp_errors
        async def resource_error_func():
            raise ResourceError("Resource not found")

        result = await resource_error_func()
        assert result["success"] is False
        assert result["error"] == "Resource not found"
        assert (
            result["error_type"] == "ResourceError"
        )  # Default error_code is the class name

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_validation_error(self):
        """Test handle_mcp_errors with ValidationError."""

        @handle_mcp_errors
        async def validation_error_func():
            raise ValidationError("Invalid input format")

        result = await validation_error_func()
        assert result["success"] is False
        assert result["error"] == "Invalid input format"
        assert result["error_type"] == "validation"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_rate_limit_error(self):
        """Test handle_mcp_errors with RateLimitError."""

        @handle_mcp_errors
        async def rate_limit_func():
            raise RateLimitError("Rate limit exceeded", retry_after=60)

        result = await rate_limit_func()
        assert result["success"] is False
        assert result["error"] == "Rate limit exceeded"
        assert result["error_type"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_handle_mcp_errors_unexpected_error(self):
        """Test handle_mcp_errors with unexpected error."""

        @handle_mcp_errors
        async def unexpected_error_func():
            raise RuntimeError("Unexpected system error")

        result = await unexpected_error_func()
        assert result["success"] is False
        assert result["error"] == "Internal server error"
        assert result["error_type"] == "internal"


class TestValidateInput:
    """Test cases for validate_input decorator."""

    @pytest.mark.asyncio
    async def test_validate_input_success(self):
        """Test validate_input with successful validation."""

        def validate_positive(value):
            if value <= 0:
                raise ValueError("Must be positive")
            return value

        @validate_input(count=validate_positive)
        async def process_count(count):
            return f"Processing {count} items"

        result = await process_count(5)
        assert result == "Processing 5 items"

    @pytest.mark.asyncio
    async def test_validate_input_failure(self):
        """Test validate_input with validation failure."""

        def validate_email(email):
            if "@" not in email:
                raise ValueError("Invalid email format")
            return email

        @validate_input(email=validate_email)
        async def send_email(email, message):
            return f"Sent '{message}' to {email}"

        with pytest.raises(ValidationError) as exc_info:
            await send_email("invalid-email", "Hello")

        error = exc_info.value
        assert "Invalid email: Invalid email format" in error.message
        assert error.context["field"] == "email"

    @pytest.mark.asyncio
    async def test_validate_input_transform_value(self):
        """Test validate_input transforms validated values."""

        def normalize_name(name):
            return name.strip().title()

        @validate_input(name=normalize_name)
        async def greet_user(name):
            return f"Hello, {name}!"

        result = await greet_user("  john doe  ")
        assert result == "Hello, John Doe!"

    @pytest.mark.asyncio
    async def test_validate_input_multiple_validators(self):
        """Test validate_input with multiple validators."""

        def validate_positive(value):
            if value <= 0:
                raise ValueError("Must be positive")
            return value

        def validate_string(value):
            if not isinstance(value, str):
                raise ValueError("Must be string")
            return value.strip()

        @validate_input(count=validate_positive, name=validate_string)
        async def process_items(count, name):
            return f"Processing {count} items for {name}"

        result = await process_items(3, "  Alice  ")
        assert result == "Processing 3 items for Alice"


# NOTE: RateLimiter and global rate limiter tests have been removed.
# Rate limiting functionality has been consolidated to use the advanced
# RateLimitManager from src.services.utilities.rate_limiter.py.
# See test_rate_limiter.py for comprehensive rate limiting tests.


class TestCreateValidationError:
    """Test cases for create_validation_error utility."""

    def test_create_validation_error_basic(self):
        """Test create_validation_error with basic parameters."""
        error = create_validation_error(
            field="email", message="Invalid email format", error_type="value_error"
        )

        # Should return a PydanticCustomError
        assert hasattr(error, "message_template")

    def test_create_validation_error_with_context(self):
        """Test create_validation_error with additional context."""
        error = create_validation_error(
            field="age",
            message="Age must be positive",
            error_type="value_error",
            value=-5,
            min_value=0,
        )

        # Should include context in the error
        assert hasattr(error, "message_template")


class TestErrorIntegration:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_hierarchy_integration(self):
        """Test error hierarchy works correctly in practice."""
        # Create a service error
        service_error = QdrantServiceError(
            "Connection failed",
            error_code="CONN_FAILED",
            context={"host": "localhost", "port": 6333},
        )

        # Should be catchable as BaseError
        assert isinstance(service_error, BaseError)
        assert isinstance(service_error, ServiceError)

        # Should serialize correctly
        error_dict = service_error.to_dict()
        assert error_dict["error_type"] == "QdrantServiceError"
        assert error_dict["context"]["host"] == "localhost"

    @pytest.mark.asyncio
    async def test_decorator_combination(self):
        """Test combining multiple error handling decorators."""

        @handle_mcp_errors
        @retry_async(max_attempts=2, base_delay=0.01)
        async def complex_operation(should_fail=False):
            if should_fail:
                raise NetworkError("Connection failed")
            return {"data": "success"}

        # Success case
        result = await complex_operation(False)
        assert result["success"] is True

        # Failure case with retry
        with patch("asyncio.sleep"):
            result = await complex_operation(True)

        assert result["success"] is False
        assert result["error_type"] == "network"

    @pytest.mark.asyncio
    @patch("src.services.errors.logger")
    async def test_error_logging_integration(self, mock_logger):
        """Test error logging works correctly."""

        @handle_mcp_errors
        async def logging_test():
            raise ToolError("Test tool error")

        result = await logging_test()

        # Should log the error
        mock_logger.warning.assert_called_once()
        log_call = mock_logger.warning.call_args[0][0]
        assert "MCP error" in log_call
        assert "Test tool error" in log_call

        # Should return safe response
        assert result["success"] is False
        assert result["error"] == "Test tool error"
