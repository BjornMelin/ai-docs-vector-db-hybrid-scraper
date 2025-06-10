"""Compression middleware for production-grade response compression.

This middleware provides intelligent response compression with configurable
thresholds and compression levels for optimal performance.
"""

import gzip
import logging
from collections.abc import Callable
from typing import ClassVar

from src.config.fastapi import CompressionConfig
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Intelligent compression middleware with content-type awareness.

    Features:
    - Configurable compression levels and thresholds
    - Content-type filtering for appropriate compression
    - Memory-efficient streaming compression
    - Automatic compression negotiation
    """

    # Content types that benefit from compression
    COMPRESSIBLE_TYPES: ClassVar[set[str]] = {
        "text/",
        "application/json",
        "application/javascript",
        "application/xml",
        "application/xhtml+xml",
        "application/rss+xml",
        "application/atom+xml",
        "image/svg+xml",
    }

    # Content types to never compress
    INCOMPRESSIBLE_TYPES: ClassVar[set[str]] = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "video/",
        "audio/",
        "application/zip",
        "application/gzip",
        "application/pdf",
    }

    def __init__(self, app: Callable, config: CompressionConfig):
        """Initialize compression middleware.

        Args:
            app: ASGI application
            config: Compression configuration
        """
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with intelligent compression."""
        if not self.config.enabled:
            return await call_next(request)

        # Check if client accepts gzip compression
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return await call_next(request)

        # Process the request
        response = await call_next(request)

        # Determine if response should be compressed
        if not self._should_compress(response):
            return response

        # Compress the response
        return await self._compress_response(response)

    def _should_compress(self, response: Response) -> bool:
        """Determine if response should be compressed.

        Args:
            response: HTTP response

        Returns:
            True if response should be compressed
        """
        # Don't compress if already compressed
        if "content-encoding" in response.headers:
            return False

        # Check content type
        content_type = response.headers.get("content-type", "")
        if not content_type:
            return False

        # Check if content type is incompressible
        for incompressible in self.INCOMPRESSIBLE_TYPES:
            if content_type.startswith(incompressible):
                return False

        # Check if content type is compressible
        is_compressible = any(
            content_type.startswith(compressible)
            for compressible in self.COMPRESSIBLE_TYPES
        )

        if not is_compressible:
            return False

        # Check content length
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length < self.config.minimum_size:
                    return False
            except ValueError:
                pass

        return True

    async def _compress_response(self, response: Response) -> Response:
        """Compress response content.

        Args:
            response: Original response

        Returns:
            Compressed response
        """
        try:
            # Handle streaming responses
            if isinstance(response, StreamingResponse):
                return await self._compress_streaming_response(response)

            # Handle regular responses
            return await self._compress_regular_response(response)

        except Exception as e:
            logger.warning(f"Compression failed: {e}, serving uncompressed")
            return response

    async def _compress_regular_response(self, response: Response) -> Response:
        """Compress regular response content.

        Args:
            response: Original response

        Returns:
            Compressed response
        """
        # Get response body
        if hasattr(response, "body"):
            body = response.body
        else:
            # For responses without body attribute, we need to render it
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

        # Skip compression if body is too small
        if len(body) < self.config.minimum_size:
            return response

        # Compress the body
        compressed_body = gzip.compress(
            body, compresslevel=self.config.compression_level
        )

        # Check if compression is beneficial
        compression_ratio = len(compressed_body) / len(body)
        if compression_ratio > 0.9:  # Less than 10% reduction
            return response

        # Create new response with compressed content
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Update headers for compression
        compressed_response.headers["content-encoding"] = "gzip"
        compressed_response.headers["content-length"] = str(len(compressed_body))
        compressed_response.headers["vary"] = "Accept-Encoding"

        # Log compression stats
        logger.debug(
            "Response compressed",
            extra={
                "original_size": len(body),
                "compressed_size": len(compressed_body),
                "compression_ratio": f"{compression_ratio:.2f}",
                "compression_level": self.config.compression_level,
            },
        )

        return compressed_response

    async def _compress_streaming_response(
        self, response: StreamingResponse
    ) -> StreamingResponse:
        """Compress streaming response content.

        Args:
            response: Original streaming response

        Returns:
            Compressed streaming response
        """

        async def compress_stream():
            compressor = gzip.GzipFile(
                mode="wb", compresslevel=self.config.compression_level
            )

            try:
                async for chunk in response.body_iterator:
                    if chunk:
                        compressed_chunk = compressor.compress(chunk)
                        if compressed_chunk:
                            yield compressed_chunk

                # Finalize compression
                final_chunk = compressor.flush()
                if final_chunk:
                    yield final_chunk

            finally:
                compressor.close()

        # Create compressed streaming response
        compressed_response = StreamingResponse(
            compress_stream(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Update headers for compression
        compressed_response.headers["content-encoding"] = "gzip"
        compressed_response.headers["vary"] = "Accept-Encoding"

        # Remove content-length as it's not accurate for streaming compression
        if "content-length" in compressed_response.headers:
            del compressed_response.headers["content-length"]

        return compressed_response


# Export middleware class
__all__ = ["CompressionMiddleware"]
