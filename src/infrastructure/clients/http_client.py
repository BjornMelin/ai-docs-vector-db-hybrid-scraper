"""HTTP client provider."""

import logging
from typing import Any

import aiohttp


logger = logging.getLogger(__name__)


class HTTPClientProvider:
    """Provider for HTTP client with health checks and session management."""

    def __init__(
        self,
        http_client: aiohttp.ClientSession,
    ):
        """Initialize the HTTP client provider."""

        self._client = http_client
        self._healthy = True

    @property
    def client(self) -> aiohttp.ClientSession | None:
        """Get the HTTP client if available and healthy."""

        if not self._healthy:
            return None
        return self._client

    async def health_check(self) -> bool:
        """Check HTTP client health."""
        try:
            if not self._client:
                return False

            # Check if session is not closed
            if self._client.closed:
                self._healthy = False
                return False

        except (AttributeError, RuntimeError, ValueError) as e:
            logger.warning("HTTP client health check failed: %s", e)
            self._healthy = False
            return False

        # If no exceptions, consider healthy
        self._healthy = True
        return True

    async def get(
        self, url: str, headers: dict[str, str] | None = None, **kwargs
    ) -> aiohttp.ClientResponse:
        """Make GET request.

        Args:
            url: Request URL
            headers: Request headers
            **kwargs: Additional parameters

        Returns:
            HTTP response
        """

        if not self.client:
            msg = "HTTP client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.get(url, headers=headers, **kwargs)

    async def post(
        self,
        url: str,
        data: Any | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> aiohttp.ClientResponse:
        """Make POST request.

        Args:
            url: Request URL
            data: Form data
            json: JSON data
            headers: Request headers
            **kwargs: Additional parameters

        Returns:
            HTTP response
        """

        if not self.client:
            msg = "HTTP client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.post(
            url, data=data, json=json, headers=headers, **kwargs
        )

    async def put(
        self,
        url: str,
        data: Any | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> aiohttp.ClientResponse:
        """Make PUT request.

        Args:
            url: Request URL
            data: Form data
            json: JSON data
            headers: Request headers
            **kwargs: Additional parameters

        Returns:
            HTTP response
        """

        if not self.client:
            msg = "HTTP client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.put(
            url, data=data, json=json, headers=headers, **kwargs
        )

    async def delete(
        self, url: str, headers: dict[str, str] | None = None, **kwargs
    ) -> aiohttp.ClientResponse:
        """Make DELETE request.

        Args:
            url: Request URL
            headers: Request headers
            **kwargs: Additional parameters

        Returns:
            HTTP response
        """

        if not self.client:
            msg = "HTTP client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.delete(url, headers=headers, **kwargs)

    async def request(
        self, method: str, url: str, headers: dict[str, str] | None = None, **kwargs
    ) -> aiohttp.ClientResponse:
        """Make generic HTTP request.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            **kwargs: Additional parameters

        Returns:
            HTTP response
        """

        if not self.client:
            msg = "HTTP client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.request(method, url, headers=headers, **kwargs)
