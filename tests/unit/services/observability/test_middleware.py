"""Tests for FastAPI observability middleware."""

import asyncio
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from src.services.observability.middleware import FastAPIObservabilityMiddleware


class TestFastAPIObservabilityMiddleware:
    """Test FastAPI observability middleware functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.app = FastAPI()
        self.mock_tracer = MagicMock()
        self.mock_meter = MagicMock()

        # Setup mock metrics
        self.mock_histogram = MagicMock()
        self.mock_counter = MagicMock()
        self.mock_up_down_counter = MagicMock()

        self.mock_meter.create_histogram.return_value = self.mock_histogram
        self.mock_meter.create_counter.return_value = self.mock_counter
        self.mock_meter.create_up_down_counter.return_value = self.mock_up_down_counter

    @patch("src.services.observability.middleware.get_tracer")
    def test_middleware_initialization(self, mock_get_tracer):
        """Test middleware initialization."""
        mock_get_tracer.return_value = self.mock_tracer

        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            service_name="test-service",
            record_request_metrics=True,
            record_ai_context=True,
        )

        assert middleware.service_name == "test-service"
        assert middleware.record_request_metrics is True
        assert middleware.record_ai_context is True
        assert middleware.tracer == self.mock_tracer

        mock_get_tracer.assert_called_once_with("test-service.middleware")

    @patch("src.services.observability.middleware.get_tracer")
    @patch("src.services.observability.middleware.get_meter")
    def test_middleware_metrics_initialization(self, mock_get_meter, mock_get_tracer):
        """Test middleware metrics initialization."""
        mock_get_tracer.return_value = self.mock_tracer
        mock_get_meter.return_value = self.mock_meter

        FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=True,
        )

        # Verify metrics were created
        assert self.mock_meter.create_histogram.call_count == 1
        assert self.mock_meter.create_counter.call_count == 1
        assert self.mock_meter.create_up_down_counter.call_count == 1

    @patch("src.services.observability.middleware.get_tracer")
    def test_middleware_metrics_initialization_failure(self, mock_get_tracer):
        """Test middleware handles metrics initialization failure."""
        mock_get_tracer.return_value = self.mock_tracer

        with patch("src.services.observability.middleware.get_meter") as mock_get_meter:
            mock_get_meter.side_effect = Exception("Meter initialization failed")

            middleware = FastAPIObservabilityMiddleware(
                app=self.app,
                record_request_metrics=True,
            )

            # Should handle exception gracefully
            assert middleware.meter is None

    @pytest.mark.asyncio
    @patch("src.services.observability.middleware.get_tracer")
    async def test_dispatch_basic_request(self, mock_get_tracer):
        """Test basic request processing."""
        mock_get_tracer.return_value = self.mock_tracer

        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=False,  # Disable metrics for simplicity
        )

        # Create mock request
        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.scheme = "http"
        request.url.hostname = "localhost"
        request.headers.get.return_value = "test-agent"
        request.query_params = {}

        # Create mock response
        async def mock_call_next(req):
            response = MagicMock(spec=Response)
            response.status_code = 200
            return response

        response = await middleware.dispatch(request, mock_call_next)

        assert response.status_code == 200

        # Verify span attributes were set
        mock_span.set_attribute.assert_any_call("http.method", "GET")
        mock_span.set_attribute.assert_any_call("http.target", "/test")
        mock_span.set_attribute.assert_any_call("service.name", "ai-docs-vector-db")

    @pytest.mark.asyncio
    @patch("src.services.observability.middleware.get_tracer")
    async def test_dispatch_with_correlation_id(self, mock_get_tracer):
        """Test request processing with correlation ID."""
        mock_get_tracer.return_value = self.mock_tracer

        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=False,
        )

        # Create mock request with correlation ID
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/search"
        request.url.scheme = "https"
        request.url.hostname = "api.example.com"
        request.headers.get.return_value = "test-agent"
        request.query_params = {}
        request.state.correlation_id = "test-correlation-123"

        async def mock_call_next(req):
            response = MagicMock(spec=Response)
            response.status_code = 201
            return response

        await middleware.dispatch(request, mock_call_next)

        # Verify correlation ID was added
        mock_span.set_attribute.assert_any_call(
            "correlation.id", "test-correlation-123"
        )

    @pytest.mark.asyncio
    @patch("src.services.observability.middleware.get_tracer")
    async def test_dispatch_with_ai_context(self, mock_get_tracer):
        """Test request processing with AI context detection."""
        mock_get_tracer.return_value = self.mock_tracer

        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_ai_context=True,
            record_request_metrics=False,
        )

        # Create mock request for AI operation
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/search"
        request.url.scheme = "https"
        request.url.hostname = "api.example.com"
        request.headers.get.return_value = "test-agent"
        request.query_params = {"model": "text-embedding-3-small", "provider": "openai"}

        async def mock_call_next(req):
            response = MagicMock(spec=Response)
            response.status_code = 200
            return response

        await middleware.dispatch(request, mock_call_next)

        # Verify AI context attributes
        mock_span.set_attribute.assert_any_call("ai.operation.type", "search")
        mock_span.set_attribute.assert_any_call(
            "ai.operation.category", "vector_search"
        )
        mock_span.set_attribute.assert_any_call("ai.model", "text-embedding-3-small")
        mock_span.set_attribute.assert_any_call("ai.provider", "openai")

    @pytest.mark.asyncio
    @patch("src.services.observability.middleware.get_tracer")
    async def test_dispatch_error_handling(self, mock_get_tracer):
        """Test request processing with exception handling."""
        mock_get_tracer.return_value = self.mock_tracer

        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=False,
        )

        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/error"
        request.url.scheme = "http"
        request.url.hostname = "localhost"
        request.headers.get.return_value = "test-agent"
        request.query_params = {}

        async def mock_call_next(req):
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await middleware.dispatch(request, mock_call_next)

        # Verify error was recorded
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.observability.middleware.get_tracer")
    @patch("src.services.observability.middleware.get_meter")
    async def test_dispatch_with_metrics(self, mock_get_meter, mock_get_tracer):
        """Test request processing with metrics recording."""
        mock_get_tracer.return_value = self.mock_tracer
        mock_get_meter.return_value = self.mock_meter

        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=True,
        )

        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.scheme = "http"
        request.url.hostname = "localhost"
        request.headers.get.return_value = "test-agent"
        request.query_params = {}

        async def mock_call_next(req):
            # Simulate some processing time
            await asyncio.sleep(0.001)
            response = MagicMock(spec=Response)
            response.status_code = 200
            return response

        await middleware.dispatch(request, mock_call_next)

        # Verify metrics were recorded
        self.mock_up_down_counter.add.assert_any_call(1)  # Active request increment
        self.mock_up_down_counter.add.assert_any_call(-1)  # Active request decrement
        self.mock_histogram.record.assert_called_once()  # Duration
        self.mock_counter.add.assert_called_once()  # Request count

    @pytest.mark.asyncio
    @patch("src.services.observability.middleware.get_tracer")
    async def test_dispatch_client_error_status(self, mock_get_tracer):
        """Test request processing with client error status."""
        mock_get_tracer.return_value = self.mock_tracer

        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=False,
        )

        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/notfound"
        request.url.scheme = "http"
        request.url.hostname = "localhost"
        request.headers.get.return_value = "test-agent"
        request.query_params = {}

        async def mock_call_next(req):
            response = MagicMock(spec=Response)
            response.status_code = 404
            return response

        await middleware.dispatch(request, mock_call_next)

        # Verify error status was set
        mock_span.set_status.assert_called_once()

    def test_add_ai_context_search_operation(self):
        """Test AI context detection for search operations."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_ai_context=True,
        )

        request = MagicMock(spec=Request)
        request.url.path = "/api/search"
        request.query_params = {"query": "test query", "model": "embedding-model"}

        span = MagicMock()

        middleware._add_ai_context(request, span)

        span.set_attribute.assert_any_call("ai.operation.type", "search")
        span.set_attribute.assert_any_call("ai.operation.category", "vector_search")
        span.set_attribute.assert_any_call("ai.model", "embedding-model")
        span.set_attribute.assert_any_call("ai.query.length", 10)

    def test_add_ai_context_embedding_operation(self):
        """Test AI context detection for embedding operations."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_ai_context=True,
        )

        request = MagicMock(spec=Request)
        request.url.path = "/api/embed"
        request.query_params = {"provider": "fastembed"}
        request.headers.get.return_value = "application/json"

        span = MagicMock()

        middleware._add_ai_context(request, span)

        span.set_attribute.assert_any_call("ai.operation.type", "embedding")
        span.set_attribute.assert_any_call("ai.operation.category", "text_embedding")
        span.set_attribute.assert_any_call("ai.provider", "fastembed")
        span.set_attribute.assert_any_call(
            "http.request.content_type", "application/json"
        )

    def test_add_ai_context_rag_operation(self):
        """Test AI context detection for RAG operations."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_ai_context=True,
        )

        request = MagicMock(spec=Request)
        request.url.path = "/api/rag/generate"
        request.query_params = {}

        span = MagicMock()

        middleware._add_ai_context(request, span)

        span.set_attribute.assert_any_call("ai.operation.type", "generation")
        span.set_attribute.assert_any_call("ai.operation.category", "rag")

    def test_add_ai_context_crawling_operation(self):
        """Test AI context detection for crawling operations."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_ai_context=True,
        )

        request = MagicMock(spec=Request)
        request.url.path = "/api/crawl"
        request.query_params = {}

        span = MagicMock()

        middleware._add_ai_context(request, span)

        span.set_attribute.assert_any_call("ai.operation.type", "crawling")
        span.set_attribute.assert_any_call("ai.operation.category", "web_scraping")

    def test_add_ai_context_exception_handling(self):
        """Test AI context addition handles exceptions gracefully."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_ai_context=True,
        )

        request = MagicMock(spec=Request)
        request.url.path.lower.side_effect = Exception("URL error")

        span = MagicMock()

        # Should not raise exception
        middleware._add_ai_context(request, span)

    def test_record_request_metrics_success(self):
        """Test successful request metrics recording."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=True,
        )
        middleware.request_duration = self.mock_histogram
        middleware.request_counter = self.mock_counter

        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"

        response = MagicMock(spec=Response)
        response.status_code = 200

        start_time = time.perf_counter() - 0.1  # 100ms ago

        middleware._record_request_metrics(request, response, start_time)

        # Verify metrics were recorded with correct attributes
        expected_attrs = {
            "method": "GET",
            "status_code": "200",
            "endpoint": "/test",
            "status_category": "success",
        }

        self.mock_histogram.record.assert_called_once()
        duration_call = self.mock_histogram.record.call_args
        assert duration_call[0][1] == expected_attrs  # Check attributes
        assert 0.05 < duration_call[0][0] < 0.15  # Check duration is reasonable

        self.mock_counter.add.assert_called_once_with(1, expected_attrs)

    def test_record_request_metrics_client_error(self):
        """Test request metrics recording for client error."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=True,
        )
        middleware.request_duration = self.mock_histogram
        middleware.request_counter = self.mock_counter

        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/test"

        response = MagicMock(spec=Response)
        response.status_code = 400

        start_time = time.perf_counter()

        middleware._record_request_metrics(request, response, start_time)

        # Verify client error category
        expected_attrs = {
            "method": "POST",
            "status_code": "400",
            "endpoint": "/api/test",
            "status_category": "client_error",
        }

        self.mock_counter.add.assert_called_once_with(1, expected_attrs)

    def test_record_request_metrics_server_error(self):
        """Test request metrics recording for server error."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=True,
        )
        middleware.request_duration = self.mock_histogram
        middleware.request_counter = self.mock_counter

        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/error"

        response = MagicMock(spec=Response)
        response.status_code = 500

        start_time = time.perf_counter()

        middleware._record_request_metrics(request, response, start_time)

        # Verify server error category
        expected_attrs = {
            "method": "GET",
            "status_code": "500",
            "endpoint": "/error",
            "status_category": "server_error",
        }

        self.mock_counter.add.assert_called_once_with(1, expected_attrs)

    def test_record_error_metrics(self):
        """Test error metrics recording."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=True,
        )
        middleware.request_duration = self.mock_histogram
        middleware.request_counter = self.mock_counter

        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/error"

        error = ValueError("Test error")
        start_time = time.perf_counter()

        middleware._record_error_metrics(request, error, start_time)

        # Verify error metrics
        expected_attrs = {
            "method": "POST",
            "status_code": "500",
            "status_category": "server_error",
            "endpoint": "/api/error",
            "error_type": "ValueError",
        }

        self.mock_counter.add.assert_called_once_with(1, expected_attrs)
        self.mock_histogram.record.assert_called_once()

    def test_record_metrics_exception_handling(self):
        """Test metrics recording handles exceptions gracefully."""
        middleware = FastAPIObservabilityMiddleware(
            app=self.app,
            record_request_metrics=True,
        )
        middleware.request_duration = MagicMock()
        middleware.request_duration.record.side_effect = Exception("Metrics error")
        middleware.request_counter = self.mock_counter

        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"

        response = MagicMock(spec=Response)
        response.status_code = 200

        start_time = time.perf_counter()

        # Should not raise exception
        middleware._record_request_metrics(request, response, start_time)
