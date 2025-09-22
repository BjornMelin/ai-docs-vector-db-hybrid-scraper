"""Example of modern async test patterns with respx.

This module demonstrates best practices for async testing in 2025,
including proper use of respx for HTTP mocking, async context managers,
and reusable test utilities.
"""

import asyncio
from typing import Any

import httpx
import pytest
import respx


class TestModernAsyncPatterns:
    """Demonstrate modern async testing patterns."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_respx_usage(self):
        """Basic respx usage for HTTP mocking."""
        # Mock a simple GET request
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(
                200, json={"status": "success", "items": [1, 2, 3]}
            )
        )

        # Make the request
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com/data")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["items"]) == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_multiple_endpoints(self):
        """Mock multiple endpoints with respx."""
        # Setup multiple mocks
        respx.get("https://api.example.com/users").mock(
            return_value=httpx.Response(
                200, json=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
            )
        )

        respx.post("https://api.example.com/users").mock(
            return_value=httpx.Response(
                201, json={"id": 3, "name": "Charlie"}, headers={"Location": "/users/3"}
            )
        )

        respx.get("https://api.example.com/users/1").mock(
            return_value=httpx.Response(
                200, json={"id": 1, "name": "Alice", "email": "alice@example.com"}
            )
        )

        # Test GET list
        async with httpx.AsyncClient() as client:
            # List users
            response = await client.get("https://api.example.com/users")
            users = response.json()
            assert len(users) == 2

            # Create user
            response = await client.post(
                "https://api.example.com/users", json={"name": "Charlie"}
            )
            assert response.status_code == 201
            assert response.headers["Location"] == "/users/3"

            # Get specific user
            response = await client.get("https://api.example.com/users/1")
            user = response.json()
            assert user["email"] == "alice@example.com"

    @respx.mock
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling with respx."""
        # Mock various error conditions
        respx.get("https://api.example.com/not-found").mock(
            return_value=httpx.Response(404, text="Not Found")
        )

        respx.get("https://api.example.com/server-error").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        respx.get("https://api.example.com/timeout").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        async with httpx.AsyncClient() as client:
            # Test 404
            response = await client.get("https://api.example.com/not-found")
            assert response.status_code == 404

            # Test 500
            response = await client.get("https://api.example.com/server-error")
            assert response.status_code == 500
            assert response.json()["error"] == "Internal Server Error"

            # Test timeout
            with pytest.raises(httpx.TimeoutException):
                await client.get("https://api.example.com/timeout")

    @pytest.mark.asyncio
    async def test_async_context_managers(self):
        """Demonstrate proper async context manager usage."""

        class AsyncService:
            def __init__(self):
                self.initialized = False
                self.closed = False
                self.operations = []

            async def __aenter__(self):
                self.initialized = True
                await asyncio.sleep(0.01)  # Simulate async init
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.closed = True
                await asyncio.sleep(0.01)  # Simulate async cleanup

            async def perform_operation(self, name: str) -> str:
                if not self.initialized:
                    error_msg = "Service not initialized"
                    raise RuntimeError(error_msg)
                self.operations.append(name)
                return f"Completed: {name}"

        # Use async context manager
        async with AsyncService() as service:
            assert service.initialized
            assert not service.closed

            result = await service.perform_operation("test_op")
            assert result == "Completed: test_op"

        # Service should be closed after context
        assert service.closed
        assert service.operations == ["test_op"]

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent async operations with proper patterns."""
        results = []

        async def async_operation(op_id: int, delay: float) -> dict[str, Any]:
            await asyncio.sleep(delay)
            return {"id": op_id, "delay": delay, "completed": True}

        # Run operations concurrently
        tasks = [
            async_operation(1, 0.05),
            async_operation(2, 0.02),
            async_operation(3, 0.03),
        ]

        # Use asyncio.gather for concurrent execution
        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert len(results) == 3
        assert all(r["completed"] for r in results)

        # Results should be in task order, not completion order
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2
        assert results[2]["id"] == 3

    @pytest.mark.asyncio
    async def test_async_http_helper(self, async_http_helper):
        """Test using the async HTTP helper utility."""
        async with async_http_helper.mock_http_context():
            # Setup mocks
            async_http_helper.setup_success_response(
                "https://api.example.com/data",
                '{"result": "success"}',
                headers={"X-Custom": "test"},
            )

            async_http_helper.setup_error_response(
                "https://api.example.com/error",
                status_code=503,
                error_message="Service Unavailable",
            )

            # Make requests
            async with httpx.AsyncClient() as client:
                # Success case
                response = await client.get("https://api.example.com/data")
                assert response.status_code == 200
                assert response.json()["result"] == "success"

                # Error case
                response = await client.get("https://api.example.com/error")
                assert response.status_code == 503

            # Check request tracking
            assert async_http_helper.get_request_count() == 2
            assert async_http_helper.get_request_count("data") == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming responses with respx."""

        # Mock a streaming response
        async def stream_content():
            for i in range(3):
                yield f"chunk_{i}\n".encode()

        respx.get("https://api.example.com/stream").mock(
            return_value=httpx.Response(
                200, content=stream_content(), headers={"Content-Type": "text/plain"}
            )
        )

        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com/stream")
            content = await response.aread()

        assert content == b"chunk_0\nchunk_1\nchunk_2\n"

    @respx.mock
    @pytest.mark.asyncio
    async def test_request_matching(self):
        """Test advanced request matching with respx."""
        # Match by headers
        respx.get(
            "https://api.example.com/auth", headers={"Authorization": "Bearer token123"}
        ).mock(return_value=httpx.Response(200, json={"user": "alice"}))

        # Match by query params
        respx.get(
            "https://api.example.com/search", params={"q": "python", "limit": "10"}
        ).mock(return_value=httpx.Response(200, json={"results": ["item1", "item2"]}))

        # Match POST by JSON content
        respx.post(
            "https://api.example.com/data", json={"action": "create", "type": "user"}
        ).mock(return_value=httpx.Response(201, json={"id": 123}))

        async with httpx.AsyncClient() as client:
            # Test header matching
            response = await client.get(
                "https://api.example.com/auth",
                headers={"Authorization": "Bearer token123"},
            )
            assert response.json()["user"] == "alice"

            # Test query param matching
            response = await client.get(
                "https://api.example.com/search", params={"q": "python", "limit": "10"}
            )
            assert len(response.json()["results"]) == 2

            # Test JSON body matching
            response = await client.post(
                "https://api.example.com/data",
                json={"action": "create", "type": "user"},
            )
            assert response.json()["id"] == 123

    @pytest.mark.asyncio
    async def test_async_cleanup_patterns(self):
        """Demonstrate proper async cleanup patterns."""
        cleanup_called = []

        class AsyncResource:
            def __init__(self, name: str):
                self.name = name
                self.active = True

            async def cleanup(self):
                await asyncio.sleep(0.01)
                self.active = False
                cleanup_called.append(self.name)

        # Create resources
        resources = [
            AsyncResource("resource1"),
            AsyncResource("resource2"),
            AsyncResource("resource3"),
        ]

        try:
            # Use resources
            for resource in resources:
                assert resource.active
        finally:
            # Cleanup all resources concurrently
            await asyncio.gather(
                *[resource.cleanup() for resource in resources],
                return_exceptions=True,  # Don't fail if one cleanup fails
            )

        # Verify all cleaned up
        assert len(cleanup_called) == 3
        assert all(not r.active for r in resources)


class TestRespxAdvancedPatterns:
    """Advanced respx patterns for complex scenarios."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_dynamic_responses(self):
        """Test dynamic response generation with respx."""
        call_count = 0

        def dynamic_response(request):
            nonlocal call_count
            call_count += 1

            # Return different responses based on call count
            if call_count == 1:
                return httpx.Response(200, json={"attempt": 1, "status": "pending"})
            if call_count == 2:
                return httpx.Response(200, json={"attempt": 2, "status": "processing"})
            return httpx.Response(
                200, json={"attempt": call_count, "status": "completed"}
            )

        # Setup dynamic mock
        respx.get("https://api.example.com/status").mock(side_effect=dynamic_response)

        async with httpx.AsyncClient() as client:
            # First call
            response = await client.get("https://api.example.com/status")
            assert response.json()["status"] == "pending"

            # Second call
            response = await client.get("https://api.example.com/status")
            assert response.json()["status"] == "processing"

            # Third call
            response = await client.get("https://api.example.com/status")
            assert response.json()["status"] == "completed"

    @respx.mock
    @pytest.mark.asyncio
    async def test_respx_with_fixtures(self, async_http_helper):
        """Combine respx with fixture-based helpers."""
        # Use both respx directly and helper
        respx.get("https://api.example.com/direct").mock(
            return_value=httpx.Response(200, text="Direct mock")
        )

        async with async_http_helper.mock_http_context():
            async_http_helper.setup_success_response(
                "https://api.example.com/helper", "Helper mock"
            )

            async with httpx.AsyncClient() as client:
                # Both should work
                direct_response = await client.get("https://api.example.com/direct")
                helper_response = await client.get("https://api.example.com/helper")

                assert direct_response.text == "Direct mock"
                assert helper_response.text == "Helper mock"
