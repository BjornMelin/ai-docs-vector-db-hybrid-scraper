"""Compression middleware exports backed by maintained ASGI libraries."""

from __future__ import annotations

from brotli_asgi import BrotliMiddleware
from starlette.middleware.gzip import GZipMiddleware


CompressionMiddleware = GZipMiddleware
BrotliCompressionMiddleware = BrotliMiddleware


__all__ = ["BrotliCompressionMiddleware", "CompressionMiddleware"]
