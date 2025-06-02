"""Unit tests for core errors module."""

import time

from pydantic import ValidationError as PydanticValidationError
from src.services.errors import APIError
from src.services.errors import BaseError
from src.services.errors import CacheServiceError
from src.services.errors import ConfigurationError
from src.services.errors import CrawlServiceError
from src.services.errors import EmbeddingServiceError
from src.services.errors import ExternalServiceError
from src.services.errors import MCPError
from src.services.errors import NetworkError
from src.services.errors import QdrantServiceError
from src.services.errors import RateLimitError
from src.services.errors import ResourceError
from src.services.errors import ServiceError
from src.services.errors import ToolError
from src.services.errors import ValidationError
from src.services.errors import create_validation_error
from src.services.errors import safe_response


class TestBaseError:
    """Test cases for BaseError class."""

    def test_base_error_initialization(self):
        """Test BaseError initialization."""
        error = BaseError(
            "Test error", error_code="test_error", context={"key": "value"}
        )

        assert error.message == "Test error"
        assert error.error_code == "test_error"
        assert error.context == {"key": "value"}
        assert str(error) == "Test error"

    def test_base_error_default_error_code(self):
        """Test BaseError with default error code."""
        error = BaseError("Test error")

        assert error.message == "Test error"
        assert error.error_code == "BaseError"
        assert error.context == {}

    def test_base_error_to_dict(self):
        """Test BaseError to_dict method."""
        error = BaseError(
            "Test error", error_code="test_error", context={"key": "value"}
        )

        result = error.to_dict()
        expected = {
            "error": "Test error",
            "error_code": "test_error",
            "error_type": "BaseError",
            "context": {"key": "value"},
        }

        assert result == expected

    def test_base_error_inheritance(self):
        """Test BaseError inheritance from Exception."""
        error = BaseError("Test error")

        assert isinstance(error, Exception)
        assert isinstance(error, BaseError)


class TestServiceErrors:
    """Test cases for service error classes."""

    def test_service_error_inheritance(self):
        """Test ServiceError inheritance."""
        error = ServiceError("Service error")

        assert isinstance(error, BaseError)
        assert isinstance(error, ServiceError)
        assert error.error_code == "ServiceError"

    def test_qdrant_service_error(self):
        """Test QdrantServiceError."""
        error = QdrantServiceError(
            "Qdrant connection failed", error_code="connection_error"
        )

        assert isinstance(error, ServiceError)
        assert isinstance(error, QdrantServiceError)
        assert error.message == "Qdrant connection failed"
        assert error.error_code == "connection_error"

    def test_embedding_service_error(self):
        """Test EmbeddingServiceError."""
        error = EmbeddingServiceError("Embedding generation failed")

        assert isinstance(error, ServiceError)
        assert isinstance(error, EmbeddingServiceError)
        assert error.error_code == "EmbeddingServiceError"

    def test_crawl_service_error(self):
        """Test CrawlServiceError."""
        error = CrawlServiceError("Web crawling failed", context={"url": "example.com"})

        assert isinstance(error, ServiceError)
        assert isinstance(error, CrawlServiceError)
        assert error.context["url"] == "example.com"

    def test_cache_service_error(self):
        """Test CacheServiceError."""
        error = CacheServiceError("Cache operation failed")

        assert isinstance(error, ServiceError)
        assert isinstance(error, CacheServiceError)


class TestValidationError:
    """Test cases for ValidationError class."""

    def test_validation_error_basic(self):
        """Test basic ValidationError."""
        error = ValidationError("Invalid input", error_code="validation_failed")

        assert isinstance(error, BaseError)
        assert error.message == "Invalid input"
        assert error.error_code == "validation_failed"

    def test_validation_error_from_pydantic_single(self):
        """Test ValidationError creation from single Pydantic error."""
        # Create a mock Pydantic ValidationError
        pydantic_error = PydanticValidationError.from_exception_data(
            "ValidationError",
            [
                {
                    "type": "string_type",
                    "loc": ("name",),
                    "msg": "Input should be a valid string",
                    "input": 123,
                }
            ],
        )

        validation_error = ValidationError.from_pydantic(pydantic_error)

        assert validation_error.message == "Input should be a valid string"
        assert validation_error.error_code == "validation_error"
        assert validation_error.context["field"] == "name"
        assert validation_error.context["type"] == "string_type"
        assert validation_error.context["input"] == 123

    def test_validation_error_from_pydantic_multiple(self):
        """Test ValidationError creation from multiple Pydantic errors."""
        pydantic_error = PydanticValidationError.from_exception_data(
            "ValidationError",
            [
                {
                    "type": "string_type",
                    "loc": ("name",),
                    "msg": "Input should be a valid string",
                    "input": 123,
                },
                {
                    "type": "int_type",
                    "loc": ("age",),
                    "msg": "Input should be a valid integer",
                    "input": "not_an_int",
                },
            ],
        )

        validation_error = ValidationError.from_pydantic(pydantic_error)

        assert validation_error.message == "Multiple validation errors occurred"
        assert validation_error.error_code == "validation_error"
        assert "errors" in validation_error.context
        assert len(validation_error.context["errors"]) == 2


class TestMCPErrors:
    """Test cases for MCP error classes."""

    def test_mcp_error_inheritance(self):
        """Test MCPError inheritance."""
        error = MCPError("MCP error")

        assert isinstance(error, BaseError)
        assert isinstance(error, MCPError)

    def test_tool_error(self):
        """Test ToolError class."""
        error = ToolError("Tool execution failed", error_code="execution_error")

        assert isinstance(error, MCPError)
        assert isinstance(error, ToolError)
        assert error.message == "Tool execution failed"
        assert error.error_code == "execution_error"

    def test_resource_error(self):
        """Test ResourceError class."""
        error = ResourceError(
            "Resource not found", error_code="not_found", context={"resource_id": "123"}
        )

        assert isinstance(error, MCPError)
        assert isinstance(error, ResourceError)
        assert error.context["resource_id"] == "123"


class TestAPIErrors:
    """Test cases for API error classes."""

    def test_api_error_basic(self):
        """Test basic APIError."""
        error = APIError("API error", status_code=400, error_code="bad_request")

        assert isinstance(error, BaseError)
        assert isinstance(error, APIError)
        assert error.status_code == 400
        assert error.error_code == "bad_request"

    def test_api_error_without_status_code(self):
        """Test APIError without status code."""
        error = APIError("API error")

        assert error.status_code is None
        assert error.error_code == "APIError"

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        error = ExternalServiceError("External service unavailable", status_code=503)

        assert isinstance(error, APIError)
        assert isinstance(error, ExternalServiceError)
        assert error.status_code == 503

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError(
            "Rate limit exceeded", retry_after=60.0, error_code="rate_limited"
        )

        assert isinstance(error, ExternalServiceError)
        assert isinstance(error, RateLimitError)
        assert error.status_code == 429
        assert error.retry_after == 60.0
        assert error.context["retry_after"] == 60.0

    def test_rate_limit_error_without_retry_after(self):
        """Test RateLimitError without retry_after."""
        error = RateLimitError("Rate limit exceeded")

        assert error.retry_after is None
        assert "retry_after" not in error.context

    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Connection timeout", error_code="timeout")

        assert isinstance(error, ExternalServiceError)
        assert isinstance(error, NetworkError)
        assert error.status_code == 503
        assert error.error_code == "timeout"


class TestConfigurationError:
    """Test cases for ConfigurationError class."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid configuration", error_code="config_invalid")

        assert isinstance(error, BaseError)
        assert isinstance(error, ConfigurationError)
        assert error.message == "Invalid configuration"
        assert error.error_code == "config_invalid"


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_safe_response_success(self):
        """Test safe_response with success."""
        response = safe_response(True, data="test", count=5)

        assert response["success"] is True
        assert "timestamp" in response
        assert response["data"] == "test"
        assert response["count"] == 5
        assert isinstance(response["timestamp"], float)

    def test_safe_response_failure_with_string_error(self):
        """Test safe_response with string error."""
        response = safe_response(
            False, error="Something went wrong", error_type="custom"
        )

        assert response["success"] is False
        assert "timestamp" in response
        assert response["error"] == "Something went wrong"
        assert response["error_type"] == "custom"

    def test_safe_response_failure_with_exception_error(self):
        """Test safe_response with Exception error."""
        exception = ValueError("Test exception")
        response = safe_response(False, error=exception)

        assert response["success"] is False
        assert response["error"] == "Test exception"
        assert response["error_type"] == "general"

    def test_safe_response_sanitizes_sensitive_info(self):
        """Test that safe_response sanitizes sensitive information."""
        error_msg = "Failed to connect with api_key=secret123 and token=abc123"
        response = safe_response(False, error=error_msg)

        assert "api_key" not in response["error"]
        assert "token" not in response["error"]
        assert "***" in response["error"]

    def test_safe_response_sanitizes_paths(self):
        """Test that safe_response sanitizes file paths."""
        error_msg = "Error reading file /home/user/secret.txt"
        response = safe_response(False, error=error_msg)

        assert "/home/" not in response["error"]
        # Check that path is sanitized (both /home/ and secret are replaced)
        assert "/****/" in response["error"]
        assert "***" in response["error"]  # secret is replaced

    def test_safe_response_sanitizes_multiple_sensitive_fields(self):
        """Test sanitization of multiple sensitive fields."""
        error_msg = "Login failed: password=secret, api_key=key123, secret=hidden"
        response = safe_response(False, error=error_msg)

        # All sensitive words should be replaced
        for sensitive_word in ["password", "api_key", "secret"]:
            assert sensitive_word not in response["error"]
        assert response["error"].count("***") >= 3

    def test_safe_response_timestamp_is_recent(self):
        """Test that safe_response timestamp is recent."""
        before = time.time()
        response = safe_response(True)
        after = time.time()

        assert before <= response["timestamp"] <= after

    def test_create_validation_error(self):
        """Test create_validation_error function."""
        error = create_validation_error(
            "email",
            "Invalid email format",
            error_type="email_format",
            example="user@domain.com",
        )

        # Should return a PydanticCustomError
        assert hasattr(error, "type")
        assert error.type == "email_format"
        # The message template and context are internal to Pydantic
        # so we just verify the error was created successfully

    def test_create_validation_error_with_defaults(self):
        """Test create_validation_error with default error_type."""
        error = create_validation_error("username", "Username too short")

        assert hasattr(error, "type")
        assert error.type == "value_error"


class TestErrorHierarchy:
    """Test cases for error class hierarchy."""

    def test_service_error_hierarchy(self):
        """Test service error class hierarchy."""
        service_errors = [
            QdrantServiceError("test"),
            EmbeddingServiceError("test"),
            CrawlServiceError("test"),
            CacheServiceError("test"),
        ]

        for error in service_errors:
            assert isinstance(error, ServiceError)
            assert isinstance(error, BaseError)
            assert isinstance(error, Exception)

    def test_api_error_hierarchy(self):
        """Test API error class hierarchy."""
        api_errors = [
            ExternalServiceError("test"),
            RateLimitError("test"),
            NetworkError("test"),
        ]

        for error in api_errors:
            assert isinstance(error, APIError)
            assert isinstance(error, BaseError)
            assert isinstance(error, Exception)

    def test_mcp_error_hierarchy(self):
        """Test MCP error class hierarchy."""
        mcp_errors = [ToolError("test"), ResourceError("test")]

        for error in mcp_errors:
            assert isinstance(error, MCPError)
            assert isinstance(error, BaseError)
            assert isinstance(error, Exception)

    def test_error_class_names_as_default_codes(self):
        """Test that error class names are used as default error codes."""
        test_cases = [
            (BaseError("test"), "BaseError"),
            (ServiceError("test"), "ServiceError"),
            (ValidationError("test"), "ValidationError"),
            (MCPError("test"), "MCPError"),
            (APIError("test"), "APIError"),
            (ConfigurationError("test"), "ConfigurationError"),
        ]

        for error, expected_code in test_cases:
            assert error.error_code == expected_code

    def test_all_errors_have_to_dict_method(self):
        """Test that all error classes have to_dict method."""
        error_classes = [
            BaseError,
            ServiceError,
            QdrantServiceError,
            EmbeddingServiceError,
            CrawlServiceError,
            CacheServiceError,
            ValidationError,
            MCPError,
            ToolError,
            ResourceError,
            APIError,
            ExternalServiceError,
            RateLimitError,
            NetworkError,
            ConfigurationError,
        ]

        for error_class in error_classes:
            error = error_class("test message")
            result = error.to_dict()

            assert isinstance(result, dict)
            assert "error" in result
            assert "error_code" in result
            assert "error_type" in result
            assert "context" in result
            assert result["error"] == "test message"
            assert result["error_type"] == error_class.__name__
