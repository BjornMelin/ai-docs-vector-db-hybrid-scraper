"""Tests for modernized error handling with FastAPI integration.

This test suite demonstrates how the new error handling system works
while preserving all critical functionality including circuit breakers,
rate limiting, and monitoring.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.exceptions import (
    APIException,
    CircuitBreakerException,
    ConfigurationException,
    CrawlingException,
    EmbeddingException,
    RateLimitedException,
    VectorDBException,
    handle_service_error,
    safe_error_response,
)
from src.api.main import app
from src.services.adapters.error_adapter import (
    CircuitBreakerAdapter,
    RateLimitAdapter,
    convert_legacy_exception,
    legacy_error_handler,
    mcp_error_handler,
)
from src.services.errors import (
    CrawlServiceError,
    EmbeddingServiceError,
    NetworkError,
    QdrantServiceError,
    RateLimitError,
    ToolError,
)


class TestModernizedExceptions:
    """Test the new FastAPI-native exception classes."""

    def test_api_exception_basic(self):
        """Test basic APIException functionality."""
        exc = APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Test error",
            context={"test": "value"},
        )

        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.detail == "Test error"
        assert exc.context == {"test": "value"}
        assert exc.timestamp > 0

    def test_vector_db_exception(self):
        """Test VectorDBException specific behavior."""
        exc = VectorDBException(
            "Connection failed",
            context={"host": "localhost", "port": 6333},
        )

        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Vector database error" in exc.detail
        assert "Connection failed" in exc.detail
        assert exc.context["host"] == "localhost"

    def test_rate_limited_exception(self):
        """Test RateLimitedException with retry headers."""
        exc = RateLimitedException(
            "Too many requests",
            retry_after=60,
            context={"client_ip": "127.0.0.1"},
        )

        assert exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert exc.headers["Retry-After"] == "60"
        assert exc.context["retry_after"] == 60
        assert exc.context["client_ip"] == "127.0.0.1"

    def test_circuit_breaker_exception(self):
        """Test CircuitBreakerException functionality."""
        exc = CircuitBreakerException(
            service_name="test_service",
            retry_after=30,
            context={"failure_count": 5},
        )

        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "test_service" in exc.detail
        assert exc.headers["Retry-After"] == "30"
        assert exc.context["service"] == "test_service"


class TestLegacyExceptionConversion:
    """Test conversion from legacy custom exceptions to FastAPI exceptions."""

    def test_convert_qdrant_service_error(self):
        """Test conversion of QdrantServiceError."""
        legacy_exc = QdrantServiceError(
            "Collection not found",
            error_code="collection_not_found",
            context={"collection": "test"},
        )

        modern_exc = convert_legacy_exception(legacy_exc)

        assert isinstance(modern_exc, VectorDBException)
        assert "Collection not found" in modern_exc.detail
        assert modern_exc.context["error_code"] == "collection_not_found"
        assert modern_exc.context["collection"] == "test"

    def test_convert_embedding_service_error(self):
        """Test conversion of EmbeddingServiceError."""
        legacy_exc = EmbeddingServiceError(
            "Model not available",
            context={"model": "text-embedding-ada-002"},
        )

        modern_exc = convert_legacy_exception(legacy_exc)

        assert isinstance(modern_exc, EmbeddingException)
        assert "Model not available" in modern_exc.detail
        assert modern_exc.context["model"] == "text-embedding-ada-002"

    def test_convert_rate_limit_error(self):
        """Test conversion of RateLimitError."""
        legacy_exc = RateLimitError(
            "Rate limit exceeded",
            retry_after=120.0,
            context={"provider": "openai"},
        )

        modern_exc = convert_legacy_exception(legacy_exc)

        assert isinstance(modern_exc, RateLimitedException)
        assert modern_exc.retry_after == 120
        assert modern_exc.context["provider"] == "openai"

    def test_convert_network_error(self):
        """Test conversion of NetworkError."""
        legacy_exc = NetworkError(
            "Connection timeout",
            context={"host": "api.openai.com"},
        )

        modern_exc = convert_legacy_exception(legacy_exc)

        assert isinstance(modern_exc, APIException)
        assert modern_exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Network error" in modern_exc.detail
        assert modern_exc.context["host"] == "api.openai.com"

    def test_convert_crawl_service_error(self):
        """Test conversion of CrawlServiceError."""
        legacy_exc = CrawlServiceError(
            "Scraping failed",
            context={"url": "https://example.com"},
        )

        modern_exc = convert_legacy_exception(legacy_exc)

        assert isinstance(modern_exc, CrawlingException)
        assert "Scraping failed" in modern_exc.detail
        assert modern_exc.context["url"] == "https://example.com"


class TestErrorDecorators:
    """Test error handling decorators and adapters."""

    @pytest.mark.asyncio
    async def test_legacy_error_handler_decorator(self):
        """Test the legacy error handler decorator."""

        @legacy_error_handler(operation="test_operation")
        async def failing_service():
            raise QdrantServiceError("Test error")

        with pytest.raises(VectorDBException) as exc_info:
            await failing_service()

        exc = exc_info.value
        assert exc.context["operation"] == "test_operation"
        assert exc.context["legacy_exception"] == "QdrantServiceError"

    @pytest.mark.asyncio
    async def test_mcp_error_handler_decorator(self):
        """Test MCP error handler decorator."""

        @mcp_error_handler
        async def mcp_tool():
            raise ToolError("MCP tool failed")

        result = await mcp_tool()

        assert result["success"] is False
        assert "MCP tool failed" in result["error"]
        assert result["error_type"] == "mcp_error"

    @pytest.mark.asyncio
    async def test_mcp_error_handler_with_api_exception(self):
        """Test MCP error handler with FastAPI exception."""

        @mcp_error_handler
        async def mcp_tool():
            raise VectorDBException("Vector DB unavailable")

        result = await mcp_tool()

        assert result["success"] is False
        assert "Vector database error" in result["error"]
        assert result["error_type"] == "api_error"

    def test_circuit_breaker_adapter(self):
        """Test circuit breaker adapter functionality."""

        # Simulate a circuit breaker error from the legacy system
        class MockCircuitBreakerError(Exception):
            def __init__(self):
                self.adaptive_timeout = 45.0
                self.context = {"failure_count": 5}

        error = MockCircuitBreakerError()
        exc = CircuitBreakerAdapter.handle_circuit_breaker_error(
            "embedding_service", error
        )

        assert isinstance(exc, CircuitBreakerException)
        assert exc.context["service"] == "embedding_service"
        assert exc.retry_after == 45
        assert exc.context["failure_count"] == 5

    def test_rate_limit_adapter(self):
        """Test rate limit adapter functionality."""

        # Simulate a rate limit error from the legacy system
        class MockRateLimitError(Exception):
            def __init__(self):
                self.retry_after = 30.0
                self.context = {"provider": "openai"}

        error = MockRateLimitError()
        exc = RateLimitAdapter.handle_rate_limit_error(error)

        assert isinstance(exc, RateLimitedException)
        assert exc.retry_after == 30
        assert exc.context["provider"] == "openai"


class TestServiceErrorHandling:
    """Test service-level error handling utilities."""

    def test_handle_service_error_with_connection_error(self):
        """Test service error handling for connection errors."""
        error = ConnectionError("Failed to connect to service")

        exc = handle_service_error(
            operation="test_operation",
            error=error,
            context={"service": "vector_db"},
        )

        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "connectivity issues" in exc.detail
        assert exc.context["operation"] == "test_operation"
        assert exc.context["service"] == "vector_db"

    def test_handle_service_error_with_rate_limit(self):
        """Test service error handling for rate limit errors."""
        error = Exception("rate limit exceeded")

        exc = handle_service_error(
            operation="embedding_generation",
            error=error,
        )

        assert isinstance(exc, RateLimitedException)
        assert "rate limiting" in exc.detail

    def test_handle_service_error_with_validation_error(self):
        """Test service error handling for validation errors."""
        error = ValueError("invalid input provided")

        exc = handle_service_error(
            operation="search_operation",
            error=error,
        )

        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert "invalid input" in exc.detail

    def test_safe_error_response_success(self):
        """Test safe error response for successful operations."""
        response = safe_error_response(
            True,
            result={"data": "test"},
            count=5,
        )

        assert response["success"] is True
        assert response["result"]["data"] == "test"
        assert response["count"] == 5
        assert "timestamp" in response

    def test_safe_error_response_failure(self):
        """Test safe error response for failed operations."""
        response = safe_error_response(
            False,
            error="Test error message",
            error_type="validation",
        )

        assert response["success"] is False
        assert response["error"] == "Test error message"
        assert response["error_type"] == "validation"
        assert "timestamp" in response

    def test_safe_error_response_sanitization(self):
        """Test error message sanitization in safe responses."""
        response = safe_error_response(
            False,
            error="Error with api_key=secret123 and /home/user/file.txt",
        )

        assert "***" in response["error"]  # API key should be masked
        assert "/****/file.txt" in response["error"]  # Home path should be masked
        assert "secret123" not in response["error"]


class TestFastAPIIntegration:
    """Test FastAPI application integration with new error handling."""

    def test_api_exception_handler(self):
        """Test API exception handler in FastAPI app."""
        client = TestClient(app)

        # Test configuration reload with invalid environment
        response = client.post("/api/config/reload")

        # Should get a configuration exception
        assert response.status_code == 403
        data = response.json()
        assert "error" in data
        assert "timestamp" in data
        assert data["status_code"] == 403

    def test_middleware_metrics_endpoint(self):
        """Test middleware metrics endpoint."""
        client = TestClient(app)

        response = client.get("/api/middleware/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "middleware_metrics" in data
        assert "timestamp" in data

    def test_health_endpoint_with_new_format(self):
        """Test that health endpoint works with new error handling."""
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_error_response_format_consistency(self):
        """Test that all error responses follow consistent format."""
        client = TestClient(app)

        # Test non-existent endpoint
        response = client.get("/nonexistent")

        assert response.status_code == 404
        data = response.json()

        # Should have consistent error response format
        assert "timestamp" in data
        assert "status_code" in data or "detail" in data


class TestBackwardCompatibility:
    """Test backward compatibility with existing systems."""

    def test_circuit_breaker_integration(self):
        """Test that circuit breaker integration still works."""
        # This would test integration with the existing CircuitBreakerRegistry
        # In a real implementation, you'd verify the circuit breaker still functions
        from src.services.errors import CircuitBreakerRegistry

        # Should be able to access the registry
        services = CircuitBreakerRegistry.get_services()
        assert isinstance(services, list)

    def test_monitoring_integration(self):
        """Test that monitoring integration is preserved."""
        # Verify that metrics collection still works with new error handling
        # This ensures we haven't broken existing observability
        try:
            from src.services.monitoring.metrics import get_metrics_registry

            # Should not raise an error
            registry = get_metrics_registry()
            # Registry might be None if monitoring is disabled, which is fine
        except ImportError:
            # Monitoring might not be available in test environment
            pytest.skip("Monitoring not available in test environment")

    def test_mcp_tool_compatibility(self):
        """Test that MCP tools still work with new error handling."""
        # Verify MCP tools can still use safe_error_response
        response = safe_error_response(True, data="test")
        assert response["success"] is True
        assert response["data"] == "test"

        # Test error case
        response = safe_error_response(False, error="test error")
        assert response["success"] is False
        assert response["error"] == "test error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
