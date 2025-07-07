"""Modern async test utilities for HTTP testing.

This module provides reusable utilities for async testing with respx.
"""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx


if TYPE_CHECKING:
    from respx.router import MockRouter


class AsyncHTTPTestHelper:
    """Helper for async HTTP testing with respx."""

    # Exception messages
    _NO_MOCK_CONTEXT_ERROR = "Must be used within mock_http_context"

    def __init__(self):
        self.mock_router: MockRouter | None = None
        self.recorded_requests: list[httpx.Request] = []

    @asynccontextmanager
    async def mock_http_context(self):
        """Context manager for HTTP mocking."""
        with respx.mock() as mock_router:
            self.mock_router = mock_router

            # Add request recording
            @mock_router.route()
            def record_request(request):
                self.recorded_requests.append(request)
                return  # Let other routes handle the response

            yield mock_router

    def setup_success_response(
        self,
        url: str,
        content: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        """Setup a successful HTTP response."""
        if not self.mock_router:
            raise RuntimeError(self._NO_MOCK_CONTEXT_ERROR)

        response_headers = headers or {"content-type": "text/html"}
        self.mock_router.get(url).mock(
            return_value=httpx.Response(
                status_code, text=content, headers=response_headers
            )
        )

    def setup_error_response(
        self,
        url: str,
        status_code: int = 500,
        error_message: str = "Internal Server Error",
    ):
        """Setup an error HTTP response."""
        if not self.mock_router:
            raise RuntimeError(self._NO_MOCK_CONTEXT_ERROR)

        self.mock_router.get(url).mock(
            return_value=httpx.Response(status_code, text=error_message)
        )

    def setup_timeout(self, url: str):
        """Setup a timeout response."""
        if not self.mock_router:
            raise RuntimeError(self._NO_MOCK_CONTEXT_ERROR)

        self.mock_router.get(url).mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

    def get_request_count(self, url_pattern: str | None = None) -> int:
        """Get count of requests matching pattern."""
        if not url_pattern:
            return len(self.recorded_requests)

        count = 0
        for request in self.recorded_requests:
            if url_pattern in str(request.url):
                count += 1
        return count


@pytest.fixture
async def async_http_helper():
    """Provide async HTTP test helper."""
    return AsyncHTTPTestHelper()


@pytest.fixture
async def mock_successful_scrape(async_http_helper):
    """Mock a successful scrape response."""

    async def _mock_scrape(url: str, content: str):
        async with async_http_helper.mock_http_context():
            async_http_helper.setup_success_response(url, content)
            yield async_http_helper

    return _mock_scrape


@pytest.fixture
async def mock_api_responses(async_http_helper):
    """Mock multiple API responses."""

    async def _mock_responses(responses: dict[str, dict[str, Any]]):
        async with async_http_helper.mock_http_context():
            for url, response_data in responses.items():
                if response_data.get("error"):
                    async_http_helper.setup_error_response(
                        url,
                        response_data.get("status_code", 500),
                        response_data.get("error"),
                    )
                elif response_data.get("timeout"):
                    async_http_helper.setup_timeout(url)
                else:
                    async_http_helper.setup_success_response(
                        url,
                        response_data.get("content", ""),
                        response_data.get("status_code", 200),
                        response_data.get("headers"),
                    )
            yield async_http_helper

    return _mock_responses
