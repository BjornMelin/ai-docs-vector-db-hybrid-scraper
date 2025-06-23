import typing
"""Response compression middleware for production optimization.

This middleware provides gzip compression for responses to reduce bandwidth
and improve performance in production environments.
"""

import gzip
import logging
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware with configurable compression levels.

    Features:
    - Gzip compression for supported content types
    - Configurable compression levels
    - Content-length aware compression
    - Client accept-encoding header checking
    """

    def __init__(
        self,
        app: Callable,
        minimum_size: int = 500,
        compression_level: int = 6,
        compressible_types: list[str] | None = None,
    ):
        """Initialize compression middleware.

        Args:
            app: ASGI application
            minimum_size: Minimum response size to compress (bytes)
            compression_level: Gzip compression level (1-9)
            compressible_types: List of content types to compress
        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = max(1, min(9, compression_level))

        # Default compressible content types
        self.compressible_types = compressible_types or [
            "text/html",
            "text/plain",
            "text/css",
            "text/javascript",
            "application/javascript",
            "application/json",
            "application/xml",
            "text/xml",
            "application/rss+xml",
            "application/atom+xml",
            "image/svg+xml",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with response compression."""
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return await call_next(request)

        # Process the request
        response = await call_next(request)

        # Check if response should be compressed
        if not self._should_compress(response):
            return response

        # Compress the response
        return await self._compress_response(response)

    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed.

        Args:
            response: HTTP response

        Returns:
            True if response should be compressed
        """
        # Don't compress if already compressed
        if response.headers.get("content-encoding"):
            return False

        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in self.compressible_types):
            return False

        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return False

        # Don't compress streaming responses (would need chunked compression)
        return not isinstance(response, StreamingResponse)

    async def _compress_response(self, response: Response) -> Response:
        """Compress response body with gzip.

        Args:
            response: HTTP response to compress

        Returns:
            Compressed response
        """
        try:
            # Get response body
            body = response.body

            # Skip if body is too small
            if len(body) < self.minimum_size:
                return response

            # Compress the body
            compressed_body = gzip.compress(body, compresslevel=self.compression_level)

            # Create new response with compressed body
            compressed_response = Response(
                content=compressed_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

            # Update headers
            compressed_response.headers["content-encoding"] = "gzip"
            compressed_response.headers["content-length"] = str(len(compressed_body))

            # Add Vary header to indicate compression varies by encoding
            vary_header = response.headers.get("vary", "")
            if "accept-encoding" not in vary_header.lower():
                if vary_header:
                    vary_header += ", Accept-Encoding"
                else:
                    vary_header = "Accept-Encoding"
                compressed_response.headers["vary"] = vary_header

            # Calculate compression ratio for logging
            compression_ratio = len(body) / len(compressed_body)
            logger.debug(
                f"Compressed response: {len(body)} -> {len(compressed_body)} bytes "
                f"(ratio: {compression_ratio:.2f}x)"
            )

            return compressed_response

        except Exception as e:
            logger.warning(f"Failed to compress response: {e}")
            return response


class BrotliCompressionMiddleware(BaseHTTPMiddleware):
    """Brotli compression middleware for even better compression ratios.

    Note: Requires brotli library to be installed.
    Falls back to gzip if brotli is not available.
    """

    def __init__(
        self,
        app: Callable,
        minimum_size: int = 500,
        quality: int = 4,
        compressible_types: list[str] | None = None,
    ):
        """Initialize Brotli compression middleware.

        Args:
            app: ASGI application
            minimum_size: Minimum response size to compress (bytes)
            quality: Brotli compression quality (0-11)
            compressible_types: List of content types to compress
        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.quality = max(0, min(11, quality))

        # Check if brotli is available
        try:
            import brotli

            self.brotli = brotli
            self.brotli_available = True
        except ImportError:
            self.brotli = None
            self.brotli_available = False
            logger.warning("Brotli compression not available, falling back to gzip")

        # Default compressible content types
        self.compressible_types = compressible_types or [
            "text/html",
            "text/plain",
            "text/css",
            "text/javascript",
            "application/javascript",
            "application/json",
            "application/xml",
            "text/xml",
            "application/rss+xml",
            "application/atom+xml",
            "image/svg+xml",
        ]

        # Fallback to gzip compression
        if not self.brotli_available:
            self.gzip_middleware = CompressionMiddleware(
                app, minimum_size, 6, compressible_types
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with Brotli compression."""
        if not self.brotli_available:
            return await self.gzip_middleware.dispatch(request, call_next)

        # Check if client accepts brotli
        accept_encoding = request.headers.get("accept-encoding", "")
        if "br" not in accept_encoding.lower():
            # Fall back to gzip if available
            if "gzip" in accept_encoding.lower():
                return await CompressionMiddleware(
                    lambda req: call_next(req),
                    self.minimum_size,
                    6,
                    self.compressible_types,
                ).dispatch(request, call_next)
            return await call_next(request)

        # Process the request
        response = await call_next(request)

        # Check if response should be compressed
        if not self._should_compress(response):
            return response

        # Compress the response
        return await self._compress_response(response)

    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed."""
        # Don't compress if already compressed
        if response.headers.get("content-encoding"):
            return False

        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in self.compressible_types):
            return False

        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return False

        # Don't compress streaming responses
        return not isinstance(response, StreamingResponse)

    async def _compress_response(self, response: Response) -> Response:
        """Compress response body with Brotli."""
        try:
            # Get response body
            body = response.body

            # Skip if body is too small
            if len(body) < self.minimum_size:
                return response

            # Compress the body
            compressed_body = self.brotli.compress(body, quality=self.quality)

            # Create new response with compressed body
            compressed_response = Response(
                content=compressed_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

            # Update headers
            compressed_response.headers["content-encoding"] = "br"
            compressed_response.headers["content-length"] = str(len(compressed_body))

            # Add Vary header
            vary_header = response.headers.get("vary", "")
            if "accept-encoding" not in vary_header.lower():
                if vary_header:
                    vary_header += ", Accept-Encoding"
                else:
                    vary_header = "Accept-Encoding"
                compressed_response.headers["vary"] = vary_header

            # Calculate compression ratio for logging
            compression_ratio = len(body) / len(compressed_body)
            logger.debug(
                f"Brotli compressed response: {len(body)} -> {len(compressed_body)} bytes "
                f"(ratio: {compression_ratio:.2f}x)"
            )

            return compressed_response

        except Exception as e:
            logger.warning(f"Failed to compress response with Brotli: {e}")
            return response


# Export middleware classes
__all__ = ["BrotliCompressionMiddleware", "CompressionMiddleware"]
