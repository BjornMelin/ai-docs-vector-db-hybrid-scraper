"""Tests for compression middleware wrappers."""

from __future__ import annotations

import inspect

from starlette.middleware.gzip import GZipMiddleware

from src.services.fastapi.middleware.compression import (
    BrotliCompressionMiddleware,
    CompressionMiddleware,
)


def test_compression_middleware_aliases_starlette_class() -> None:
    """Test that CompressionMiddleware aliases Starlette's GZipMiddleware."""
    assert CompressionMiddleware is GZipMiddleware


def test_brotli_middleware_is_available() -> None:
    """Ensure brotli wrapper is present and callable."""
    assert inspect.isclass(BrotliCompressionMiddleware)
    # Optional dependency fallback: if brotli not installed, alias equals gzip.
    if BrotliCompressionMiddleware is CompressionMiddleware:
        assert BrotliCompressionMiddleware is GZipMiddleware
    else:
        # When brotli is installed the middleware should define default quality kwarg.
        signature = inspect.signature(BrotliCompressionMiddleware.__init__)
        assert "quality" in signature.parameters
