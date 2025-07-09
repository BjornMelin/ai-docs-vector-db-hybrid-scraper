"""Factory for creating optimized HTTPX clients with HTTP/2 and connection pooling."""

from typing import Any

import httpx


class HTTPClientFactory:
    """Factory for creating optimized HTTPX clients.

    Provides standardized configuration for HTTP/2 support and connection pooling
    across the application to improve performance and resource utilization.
    """

    @staticmethod
    def create_client(
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        max_keepalive_connections: int = 50,
        max_connections: int = 100,
        keepalive_expiry: float = 30.0,
        http2: bool = True,
        follow_redirects: bool = True,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        """Create an optimized HTTPX async client.

        Args:
            timeout: Total request timeout in seconds
            connect_timeout: Connection timeout in seconds
            max_keepalive_connections: Maximum number of keepalive connections
            max_connections: Maximum total connections
            keepalive_expiry: How long to keep connections alive
            http2: Enable HTTP/2 support
            follow_redirects: Whether to follow redirects
            headers: Additional headers to include
            **kwargs: Additional arguments passed to httpx.AsyncClient

        Returns:
            Configured httpx.AsyncClient instance
        """
        # Configure connection pooling limits
        limits = httpx.Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
            keepalive_expiry=keepalive_expiry,
        )

        # Configure timeouts
        timeout_config = httpx.Timeout(
            timeout, connect=connect_timeout, read=timeout, write=5.0, pool=2.0
        )

        # Default headers
        default_headers = {
            "User-Agent": "AI-Docs-Scraper/1.0 (+https://github.com/ai-docs-vector-db)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        if headers:
            default_headers.update(headers)

        return httpx.AsyncClient(
            limits=limits,
            timeout=timeout_config,
            http2=http2,
            follow_redirects=follow_redirects,
            headers=default_headers,
            **kwargs,
        )

    @staticmethod
    def create_lightweight_client(
        timeout: float = 10.0, **kwargs: Any
    ) -> httpx.AsyncClient:
        """Create a lightweight client for simple requests.

        Uses reduced connection pool settings suitable for occasional requests.

        Args:
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to create_client

        Returns:
            Configured httpx.AsyncClient instance
        """
        return HTTPClientFactory.create_client(
            timeout=timeout, max_keepalive_connections=20, max_connections=50, **kwargs
        )

    @staticmethod
    def create_test_client(timeout: float = 30.0, **kwargs: Any) -> httpx.AsyncClient:
        """Create a client optimized for testing.

        Uses reduced connection pool settings suitable for test environments.

        Args:
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to create_client

        Returns:
            Configured httpx.AsyncClient instance
        """
        return HTTPClientFactory.create_client(
            timeout=timeout,
            max_keepalive_connections=10,
            max_connections=20,
            headers={"User-Agent": "TestClient/1.0", "X-Test-Run": "true"},
            **kwargs,
        )
