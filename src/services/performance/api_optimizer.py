"""API Response Optimization Module.

This module provides optimizations for API responses including
compression, caching, and streaming capabilities.
"""

import gzip
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from src.services.cache.intelligent import IntelligentCache


logger = logging.getLogger(__name__)


class APIResponseOptimizer:
    """Optimize API responses for performance."""

    def __init__(self, cache: IntelligentCache):
        """Initialize API response optimizer.

        Args:
            cache: Intelligent cache instance

        """
        self.cache = cache
        self.compression_threshold = 1024  # Compress responses > 1KB
        self.streaming_threshold = 10240  # Stream responses > 10KB

    def should_compress(self, data: Any) -> bool:
        """Determine if response should be compressed.

        Args:
            data: Response data

        Returns:
            True if compression should be applied

        """
        if isinstance(data, dict | list):
            size = len(json.dumps(data))
        elif isinstance(data, str | bytes):
            size = len(data)
        else:
            return False

        return size > self.compression_threshold

    def compress_response(self, data: Any) -> bytes:
        """Compress response data using gzip.

        Args:
            data: Data to compress

        Returns:
            Compressed data

        """
        if isinstance(data, dict | list):
            data = json.dumps(data).encode("utf-8")
        elif isinstance(data, str):
            data = data.encode("utf-8")
        elif not isinstance(data, bytes):
            data = str(data).encode("utf-8")

        return gzip.compress(data, compresslevel=6)

    async def get_cached_or_compute(
        self, cache_key: str, compute_func: Any, ttl: int = 300, compress: bool = True
    ) -> Any:
        """Get cached response or compute and cache.

        Args:
            cache_key: Cache key
            compute_func: Async function to compute response
            ttl: Cache TTL in seconds
            compress: Whether to compress cached data

        Returns:
            Response data

        """
        # Check cache
        cached = await self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached

        # Compute response
        logger.debug(f"Cache miss for key: {cache_key}")
        result = await compute_func()

        # Cache result
        cache_data = result
        if compress and self.should_compress(result):
            cache_data = {
                "_compressed": True,
                "data": self.compress_response(result).hex(),
            }

        await self.cache.set(cache_key, cache_data, ttl=ttl)

        return result

    async def create_streaming_response(
        self,
        data_generator: AsyncGenerator[Any],
        media_type: str = "application/json",
    ) -> StreamingResponse:
        """Create a streaming response for large data.

        Args:
            data_generator: Async generator yielding data chunks
            media_type: Response media type

        Returns:
            StreamingResponse object

        """

        async def stream_with_compression():
            async for chunk in data_generator:
                if isinstance(chunk, dict | list):
                    processed_chunk = json.dumps(chunk) + "\n"
                    chunk_data = processed_chunk.encode("utf-8")
                elif isinstance(chunk, str):
                    chunk_data = chunk.encode("utf-8")
                else:
                    chunk_data = chunk

                # Compress chunk
                compressed = gzip.compress(chunk_data)
                yield compressed

        return StreamingResponse(
            stream_with_compression(),
            media_type=media_type,
            headers={
                "Content-Encoding": "gzip",
                "Cache-Control": "no-cache",
            },
        )

    def add_cache_headers(
        self, response: Response, max_age: int = 300, must_revalidate: bool = True
    ) -> Response:
        """Add cache control headers to response.

        Args:
            response: FastAPI response object
            max_age: Max age in seconds
            must_revalidate: Whether cache must revalidate

        Returns:
            Response with cache headers

        """
        cache_control = f"max-age={max_age}"
        if must_revalidate:
            cache_control += ", must-revalidate"

        response.headers["Cache-Control"] = cache_control
        response.headers["X-Cache-TTL"] = str(max_age)

        return response

    async def batch_requests(
        self,
        requests: list[dict[str, Any]],
        process_func: Any,
        max_batch_size: int = 100,
    ) -> list[Any]:
        """Process multiple requests in batches.

        Args:
            requests: List of request data
            process_func: Function to process batch
            max_batch_size: Maximum batch size

        Returns:
            List of results

        """
        results = []

        for i in range(0, len(requests), max_batch_size):
            batch = requests[i : i + max_batch_size]
            batch_results = await process_func(batch)
            results.extend(batch_results)

        return results
