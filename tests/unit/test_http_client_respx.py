"""Unit tests for HTTP client using respx for mocking.

This module demonstrates proper HTTP mocking with respx following
the testing best practices from CLAUDE.md.
"""

import asyncio

import httpx
import pytest
from hypothesis import given, strategies as st

from src.infrastructure.clients.http_client import HTTPClientProvider


class TestHTTPClientProviderWithRespx:
    """Test HTTP client with respx mocking."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_get_request_success(self, respx_mock):
        """Test successful GET request."""
        # Arrange
        url = "https://api.example.com/data"
        expected_response = {"status": "success", "data": [1, 2, 3]}

        respx_mock.get(url).mock(
            return_value=httpx.Response(200, json=expected_response)
        )

        client = HTTPClientProvider()
        await client.initialize()

        # Act
        response = await client.get(url)

        # Assert
        assert response == expected_response
        assert respx_mock.calls.call_count == 1

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_post_request_with_retry(self, respx_mock):
        """Test POST request with retry logic."""
        # Arrange
        url = "https://api.example.com/submit"
        payload = {"key": "value"}

        # Mock first attempt to fail, second to succeed
        route = respx_mock.post(url)
        route.side_effect = [
            httpx.Response(500, json={"error": "server error"}),
            httpx.Response(200, json={"status": "success"}),
        ]

        client = HTTPClientProvider(max_retries=2)
        await client.initialize()

        # Act
        response = await client.post(url, json=payload)

        # Assert
        assert response["status"] == "success"
        assert respx_mock.calls.call_count == 2

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_concurrent_requests(self, respx_mock):
        """Test concurrent HTTP requests."""
        # Arrange
        urls = [f"https://api.example.com/item/{i}" for i in range(5)]

        for i, url in enumerate(urls):
            respx_mock.get(url).mock(
                return_value=httpx.Response(200, json={"id": i, "data": f"item_{i}"})
            )

        client = HTTPClientProvider()
        await client.initialize()

        # Act
        responses = await asyncio.gather(*[client.get(url) for url in urls])

        # Assert
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response["id"] == i
            assert response["data"] == f"item_{i}"

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.hypothesis
    @given(
        status_code=st.sampled_from([200, 201, 204, 400, 404, 500]),
        url=st.text(min_size=10, max_size=50).map(
            lambda s: f"https://api.example.com/{s}"
        ),
    )
    async def test_various_status_codes(self, respx_mock, status_code, url):
        """Property-based test for various HTTP status codes."""
        # Arrange
        response_data = {"status": status_code}
        respx_mock.get(url).mock(
            return_value=httpx.Response(status_code, json=response_data)
        )

        client = HTTPClientProvider()
        await client.initialize()

        # Act & Assert
        if status_code < 400:
            response = await client.get(url)
            assert response == response_data
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get(url)

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_timeout_handling(self, respx_mock):
        """Test request timeout handling."""
        # Arrange
        url = "https://api.example.com/slow"

        # Mock a timeout
        respx_mock.get(url).mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        client = HTTPClientProvider(timeout=1.0)
        await client.initialize()

        # Act & Assert
        with pytest.raises(httpx.TimeoutException):
            await client.get(url)

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_custom_headers(self, respx_mock):
        """Test requests with custom headers."""
        # Arrange
        url = "https://api.example.com/protected"
        headers = {"Authorization": "Bearer test-token", "X-Custom-Header": "value"}

        # Mock route that validates headers
        def check_headers(request):
            assert request.headers["Authorization"] == "Bearer test-token"
            assert request.headers["X-Custom-Header"] == "value"
            return httpx.Response(200, json={"authenticated": True})

        respx_mock.get(url).mock(side_effect=check_headers)

        client = HTTPClientProvider()
        await client.initialize()

        # Act
        response = await client.get(url, headers=headers)

        # Assert
        assert response["authenticated"] is True

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_connection_pooling(self, respx_mock):
        """Test HTTP connection pooling."""
        # Arrange
        base_url = "https://api.example.com"
        endpoints = ["/users", "/posts", "/comments"]

        for endpoint in endpoints:
            respx_mock.get(f"{base_url}{endpoint}").mock(
                return_value=httpx.Response(200, json={"endpoint": endpoint})
            )

        client = HTTPClientProvider(base_url=base_url)
        await client.initialize()

        # Act - Multiple requests should reuse connections
        responses = []
        for _ in range(3):  # Multiple iterations
            for endpoint in endpoints:
                response = await client.get(endpoint)
                responses.append(response)

        # Assert
        assert len(responses) == 9
        assert all(r["endpoint"] in endpoints for r in responses)

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def test_http_methods(self, respx_mock, method):
        """Test various HTTP methods."""
        # Arrange
        url = "https://api.example.com/resource"
        expected_response = {"method": method, "success": True}

        # Mock the appropriate method
        route = getattr(respx_mock, method.lower())(url)
        route.mock(return_value=httpx.Response(200, json=expected_response))

        client = HTTPClientProvider()
        await client.initialize()

        # Act
        client_method = getattr(client, method.lower())
        response = await client_method(url)

        # Assert
        assert response == expected_response

        # Cleanup
        await client.cleanup()


class TestHTTPClientProviderErrorScenarios:
    """Test error scenarios with respx."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_network_error(self, respx_mock):
        """Test network error handling."""
        # Arrange
        url = "https://api.example.com/error"
        respx_mock.get(url).mock(side_effect=httpx.NetworkError("Network unreachable"))

        client = HTTPClientProvider()
        await client.initialize()

        # Act & Assert
        with pytest.raises(httpx.NetworkError):
            await client.get(url)

        # Cleanup
        await client.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_rate_limiting(self, respx_mock):
        """Test rate limiting response handling."""
        # Arrange
        url = "https://api.example.com/limited"

        respx_mock.get(url).mock(
            return_value=httpx.Response(
                429,
                json={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )
        )

        client = HTTPClientProvider()
        await client.initialize()

        # Act & Assert
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get(url)

        assert exc_info.value.response.status_code == 429
        assert exc_info.value.response.headers["Retry-After"] == "60"

        # Cleanup
        await client.cleanup()
