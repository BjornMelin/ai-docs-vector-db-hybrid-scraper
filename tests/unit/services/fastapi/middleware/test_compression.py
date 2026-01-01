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
        # When brotli is installed the middleware should accept a quality argument.
        signature = inspect.signature(BrotliCompressionMiddleware.__init__)
        params = signature.parameters
        if "quality" in params:
            assert params["quality"].default is not inspect.Signature.empty
        else:
            # Fallback to accepting arbitrary keyword arguments.
            last_param = list(params.values())[-1]
            assert last_param.kind is inspect.Parameter.VAR_KEYWORD
