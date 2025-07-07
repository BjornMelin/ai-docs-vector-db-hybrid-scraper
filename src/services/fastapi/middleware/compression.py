"""Response compression middleware for production optimization.

This middleware provides gzip compression for responses to reduce bandwidth
and improve performance in production environments.
"""

import gzip
import logging
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse


try:
    import brotli
except ImportError:
    brotli = None


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
            body = self._get_response_body(response)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to get response body: {e}")
            return response

        # Skip if body is too small
        if len(body) < self.minimum_size:
            return response

        try:
            compressed_body = gzip.compress(body, compresslevel=self.compression_level)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to compress response body: {e}")
            return response

        try:
            return self._create_compressed_response(
                response, body, compressed_body, "gzip"
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to create compressed response: {e}")
            return response

    def _get_response_body(self, response: Response) -> bytes:
        """Get response body safely.

        Args:
            response: HTTP response

        Returns:
            Response body as bytes

        Raises:
            ValueError: If body cannot be retrieved
        """
        try:
            return response.body
        except (AttributeError, UnicodeDecodeError) as e:
            error_msg = f"Cannot get response body: {e}"
            raise ValueError(error_msg) from e

    def _create_compressed_response(
        self,
        response: Response,
        original_body: bytes,
        compressed_body: bytes,
        encoding: str,
    ) -> Response:
        """Create compressed response with proper headers.

        Args:
            response: Original response
            original_body: Original response body
            compressed_body: Compressed response body
            encoding: Compression encoding (gzip or br)

        Returns:
            Compressed response
        """
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Update compression headers
        compressed_response.headers["content-encoding"] = encoding
        compressed_response.headers["content-length"] = str(len(compressed_body))

        # Add Vary header to indicate compression varies by encoding
        self._add_vary_header(compressed_response, response)

        # Log compression ratio
        compression_ratio = len(original_body) / len(compressed_body)
        logger.debug(
            f"Compressed response: {len(original_body)} -> {len(compressed_body)} bytes "
            f"(ratio: {compression_ratio:.2f}x)"
        )

        return compressed_response

    def _add_vary_header(
        self, compressed_response: Response, original_response: Response
    ) -> None:
        """Add Vary header to compressed response.

        Args:
            compressed_response: Compressed response to modify
            original_response: Original response for header reference
        """
        vary_header = original_response.headers.get("vary", "")
        if "accept-encoding" not in vary_header.lower():
            if vary_header:
                vary_header += ", Accept-Encoding"
            else:
                vary_header = "Accept-Encoding"
            compressed_response.headers["vary"] = vary_header


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
        if brotli is not None:
            self.brotli = brotli
            self.brotli_available = True
        else:
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
            body = self._get_response_body(response)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to get response body: {e}")
            return response

        # Skip if body is too small
        if len(body) < self.minimum_size:
            return response

        try:
            compressed_body = self.brotli.compress(body, quality=self.quality)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to compress response body with Brotli: {e}")
            return response

        try:
            return self._create_compressed_response(
                response, body, compressed_body, "br"
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to create Brotli compressed response: {e}")
            return response

    def _get_response_body(self, response: Response) -> bytes:
        """Get response body safely.

        Args:
            response: HTTP response

        Returns:
            Response body as bytes

        Raises:
            ValueError: If body cannot be retrieved
        """
        try:
            return response.body
        except (AttributeError, UnicodeDecodeError) as e:
            error_msg = f"Cannot get response body: {e}"
            raise ValueError(error_msg) from e

    def _create_compressed_response(
        self,
        response: Response,
        original_body: bytes,
        compressed_body: bytes,
        encoding: str,
    ) -> Response:
        """Create compressed response with proper headers.

        Args:
            response: Original response
            original_body: Original response body
            compressed_body: Compressed response body
            encoding: Compression encoding (gzip or br)

        Returns:
            Compressed response
        """
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Update compression headers
        compressed_response.headers["content-encoding"] = encoding
        compressed_response.headers["content-length"] = str(len(compressed_body))

        # Add Vary header
        self._add_vary_header(compressed_response, response)

        # Log compression ratio
        compression_ratio = len(original_body) / len(compressed_body)
        logger.debug(
            f"Brotli compressed response: {len(original_body)} -> {len(compressed_body)} bytes "
            f"(ratio: {compression_ratio:.2f}x)"
        )

        return compressed_response

    def _add_vary_header(
        self, compressed_response: Response, original_response: Response
    ) -> None:
        """Add Vary header to compressed response.

        Args:
            compressed_response: Compressed response to modify
            original_response: Original response for header reference
        """
        vary_header = original_response.headers.get("vary", "")
        if "accept-encoding" not in vary_header.lower():
            if vary_header:
                vary_header += ", Accept-Encoding"
            else:
                vary_header = "Accept-Encoding"
            compressed_response.headers["vary"] = vary_header


# Export middleware classes
__all__ = ["BrotliCompressionMiddleware", "CompressionMiddleware"]
